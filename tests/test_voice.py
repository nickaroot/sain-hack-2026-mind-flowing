"""Голос для клиента: запись вопроса превращается в текст, ответ эксперта — в mp3; ключ прокси остаётся на сервере."""

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app import config, voice
from app import llm as llm_module
from app.main import app

MP3 = b"ID3" + b"\x00" * 64
WEBM = b"\x1aE\xdf\xa3" + b"\x01" * 200
REAL_CLIENT = httpx.AsyncClient  # подмена прокси заменяет httpx.AsyncClient; запросы к самому приложению идут настоящим


@pytest.fixture
def proxy(monkeypatch):
    """Подменяет прокси: запоминает запросы и отвечает тем, что положено в .reply (статус, тело)."""

    class Proxy:
        reply = None  # (статус, тело) вместо обычного ответа
        unreachable = False

        def __init__(self):
            self.requests: list[httpx.Request] = []

    seen = Proxy()

    def respond(request: httpx.Request) -> httpx.Response:
        if seen.unreachable:
            raise httpx.ConnectError("нет сети")
        request.read()
        seen.requests.append(request)
        if seen.reply is not None:
            status, payload = seen.reply
            return httpx.Response(status, json=payload)
        if request.url.path.endswith("/audio/speech"):
            return httpx.Response(200, content=MP3, headers={"content-type": "audio/mpeg"})
        return httpx.Response(200, json={"text": " Мешает ли кофе вечером уснуть? "})

    monkeypatch.setattr(voice.httpx, "AsyncClient", lambda **kw: REAL_CLIENT(transport=httpx.MockTransport(respond)))
    monkeypatch.setattr(llm_module, "MIN_SECONDS_BETWEEN_CALLS", 0)
    monkeypatch.setattr(config, "LLM_BASE_URL", "http://proxy.test/v1")
    monkeypatch.setattr(config, "LLM_API_KEY", "test-key")
    return seen


@pytest.fixture
def client(proxy):
    app.state.runs.clear()
    app.state.voice = None
    yield TestClient(app)
    app.state.voice = None


def record(client, audio=WEBM, content_type="audio/webm;codecs=opus"):
    return client.post("/api/transcribe", content=audio, headers={"Content-Type": content_type})


def test_recording_becomes_text_and_key_stays_on_the_server(client, proxy):
    response = record(client)
    assert response.status_code == 200 and response.json() == {"text": "Мешает ли кофе вечером уснуть?"}
    sent = proxy.requests[0]
    assert sent.url.path == "/v1/audio/transcriptions" and sent.headers["authorization"] == "Bearer test-key"
    body = sent.content
    assert b'filename="question.webm"' in body and WEBM in body  # формат прокси узнаёт по расширению
    assert config.STT_MODEL.encode() in body and b'name="language"' in body


def test_answer_is_spoken_in_the_voice_of_its_expert(client, proxy):
    spoken = client.post("/api/speak", json={"text": "Да, кофе вечером мешает уснуть.", "expert_id": "biohacker"})
    assert spoken.status_code == 200 and spoken.content == MP3 and spoken.headers["content-type"] == "audio/mpeg"
    sent = json.loads(proxy.requests[0].content)
    assert sent == {"model": config.TTS_MODEL, "voice": voice.VOICES["biohacker"],
                    "input": "Да, кофе вечером мешает уснуть.", "response_format": "mp3"}
    assert len(set(voice.VOICES.values())) == 3  # три эксперта — три разных голоса
    client.post("/api/speak", json={"text": "Да, кофе вечером мешает уснуть.", "expert_id": "новый эксперт"})
    assert json.loads(proxy.requests[1].content)["voice"] == voice.DEFAULT_VOICE


def test_listening_again_does_not_spend_the_voice_budget(client, proxy):
    question = {"text": "Да, кофе вечером мешает уснуть.", "expert_id": "clinician"}
    assert client.post("/api/speak", json=question).content == client.post("/api/speak", json=question).content
    assert len(proxy.requests) == 1


@pytest.mark.parametrize("reply,status,retry_after", [
    ((402, {"error": "budget"}), 429, "3600"),  # суточный бюджет голоса у организаторов кончился
    ((429, {"error": "rate"}), 429, "60"),
    ((413, {"error": "large"}), 413, None),
    ((500, {"error": "oops"}), 502, "10"),
    ((200, {"unexpected": True}), 502, None),  # ответ не по протоколу
])
def test_proxy_failures_become_clear_answers_for_the_client(client, proxy, reply, status, retry_after):
    proxy.reply = reply
    response = record(client)
    assert response.status_code == status and response.json()["detail"]
    assert response.headers.get("Retry-After") == retry_after


def test_unreachable_proxy_is_a_502(client, proxy):
    proxy.unreachable = True
    response = record(client)
    assert response.status_code == 502 and response.headers["Retry-After"] == "10"


def test_wrong_or_empty_recording_is_rejected_without_calling_the_proxy(client, proxy):
    assert record(client, content_type="text/plain").status_code == 415
    assert record(client, audio=b"").status_code == 422
    assert client.post("/api/speak", json={"text": ""}).status_code == 422
    assert client.post("/api/speak", json={"text": "я" * (config.VOICE_MAX_TEXT_CHARS + 1)}).status_code == 422
    assert proxy.requests == []


def test_recording_may_be_large_but_not_larger_than_the_proxy_accepts(client, proxy):
    assert record(client, audio=b"\x01" * (config.API_MAX_BODY_BYTES * 4)).status_code == 200  # больше предела для текста
    assert record(client, audio=b"\x01" * (config.VOICE_MAX_AUDIO_BYTES + 1)).status_code == 413
    assert len(proxy.requests) == 1
    huge = b'{"expert_id": "biohacker", "text": "' + b"x" * (config.API_MAX_BODY_BYTES + 1) + b'"}'
    assert client.post("/api/speak", content=huge, headers={"Content-Type": "application/json"}).status_code == 413


def test_voice_waits_while_a_run_is_in_progress(client, proxy):
    class Unfinished:
        def done(self):
            return False

    app.state.runs[("code", "token")] = Unfinished()
    assert record(client).status_code == 503
    assert client.post("/api/speak", json={"text": "Да."}).status_code == 503
    app.state.runs.clear()
    assert proxy.requests == []


def test_voice_has_its_own_daily_ceiling(client, proxy, monkeypatch):
    monkeypatch.setattr(config, "VOICE_DAILY_TRANSCRIPTIONS", 1)
    monkeypatch.setattr(config, "VOICE_DAILY_SPEECH_CHARS", 10)
    assert record(client).status_code == 200 and record(client).status_code == 429
    assert client.post("/api/speak", json={"text": "Короткий."}).status_code == 200
    assert client.post("/api/speak", json={"text": "Ещё один ответ."}).status_code == 429


async def many(calls):
    """Несколько запросов к приложению одновременно, в одном цикле событий."""
    async with REAL_CLIENT(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
        return await asyncio.gather(*(call(http) for call in calls))


def speak(text, expert_id="clinician"):
    return lambda http: http.post("/api/speak", json={"text": text, "expert_id": expert_id})


def test_daily_ceiling_holds_when_requests_wait_in_the_queue_together(client, proxy, monkeypatch):
    monkeypatch.setattr(config, "VOICE_DAILY_SPEECH_CHARS", 4)
    statuses = sorted(r.status_code for r in asyncio.run(many([speak("abcd"), speak("efgh"), speak("ijkl")])))
    assert statuses == [200, 429, 429] and len(proxy.requests) == 1  # потолок проверяется и списывается под очередью
    monkeypatch.setattr(config, "VOICE_DAILY_TRANSCRIPTIONS", 1)
    app.state.voice = None  # очередь привязана к циклу событий, в котором её ждали: у второго цикла своя

    def recording(http):
        return http.post("/api/transcribe", content=WEBM, headers={"Content-Type": "audio/webm"})

    statuses = sorted(r.status_code for r in asyncio.run(many([recording] * 3)))
    assert statuses == [200, 429, 429] and len(proxy.requests) == 2


def test_requests_waiting_in_the_voice_queue_yield_to_a_run_that_started(client, proxy, monkeypatch):
    class Unfinished:
        def done(self):
            return False

    turns = []

    async def slow_turn():
        turns.append(1)
        if len(turns) == 1:  # пока первый запрос ждёт своей очереди к прокси, стенд вызывает /run
            await asyncio.sleep(0.05)
            app.state.runs[("code", "token")] = Unfinished()

    monkeypatch.setattr(voice.llm, "wait_turn", slow_turn)
    monkeypatch.setattr(voice, "PARALLEL_REQUESTS", 1)
    app.state.voice = None
    statuses = [r.status_code for r in asyncio.run(many([speak("один"), speak("два"), speak("три")]))]
    app.state.runs.clear()
    assert statuses == [200, 503, 503] and len(proxy.requests) == 1  # ушедший запрос не отзывается, ждавшие — уступают


def test_same_answer_asked_twice_at_once_is_synthesized_once(client, proxy):
    same = asyncio.run(many([speak("Да, мешает."), speak("Да, мешает."), speak("Да, мешает.", "biohacker")]))
    assert [r.status_code for r in same] == [200, 200, 200]
    assert len(proxy.requests) == 2  # один голос — один вызов, другой голос — отдельный
    assert voice.state_of(app).in_work == {} and voice.state_of(app).speech_chars == 2 * len("Да, мешает.")
    proxy.reply = (500, {"error": "oops"})
    failed = asyncio.run(many([speak("Нет."), speak("Нет.")]))
    assert [r.status_code for r in failed] == [502, 502] and len(proxy.requests) == 3
    proxy.reply = None
    assert asyncio.run(many([speak("Нет.")]))[0].status_code == 200  # сбой не запоминается: повтор удался


def test_same_recording_sent_again_is_not_transcribed_twice(client, proxy):
    assert record(client).json() == record(client).json() == {"text": "Мешает ли кофе вечером уснуть?"}
    assert len(proxy.requests) == 1 and voice.state_of(app).transcriptions == 1
    assert record(client, audio=WEBM + b"\x02").status_code == 200 and len(proxy.requests) == 2  # другая запись


def test_voice_requests_take_their_turn_in_the_shared_queue(client, proxy, monkeypatch):
    turns = []

    async def counted():
        turns.append(1)

    monkeypatch.setattr(voice.llm, "wait_turn", counted)
    record(client)
    client.post("/api/speak", json={"text": "Да."})
    assert len(turns) == 2  # у ключа один лимит запросов в минуту на нейросеть и голос


def test_voice_without_proxy_settings_is_a_clear_503(client, monkeypatch):
    monkeypatch.setattr(config, "LLM_API_KEY", "")
    assert record(client).status_code == 503
