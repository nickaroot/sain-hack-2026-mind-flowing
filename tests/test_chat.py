"""Методы для клиента: эксперты, разговор и источник. Разговор не должен мешать прогону набора."""

import asyncio

import httpx
import pytest
from fastapi.testclient import TestClient

from app import chat, config, hack_api, llm as llm_module, pipeline
from app.main import app
from tests.fakes import BODY, DATASET, EXPERTS, REFUSAL, FakeLLM, good_answer

LIVE = {**DATASET, "users": [{"user_id": "u1"}], "chat_history": [
    {"user_id": "u1", "role": "user", "at": "2026-11-10T22:00:00+03:00", "text": "Пью кофе в шесть вечера."},
]}
QUESTION = {"expert_id": "biohacker", "text": "Мешает ли кофе вечером уснуть?"}


def reset() -> None:
    app.state.dataset = None
    app.state.runs.clear()
    app.state.run_finished = False
    app.state.chat = None


@pytest.fixture
def client(monkeypatch):
    reset()
    app.state.dataset = LIVE
    monkeypatch.setattr(config, "CHAT_FREEZE_FROM", "")  # окно перед зачётным прогоном проверяется отдельным тестом
    yield TestClient(app)
    reset()


def ask(client, monkeypatch, llm, **body):
    monkeypatch.setattr(pipeline, "LLM", lambda: llm)
    return client.post("/api/chat", json={**QUESTION, **body})


class SlowLLM(FakeLLM):
    """Отвечает не сразу: за это время успевают прийти другие запросы."""

    def __init__(self, *answers, seconds=0.05, tokens=0):
        super().__init__(*answers)
        self.seconds, self.tokens, self.started = seconds, tokens, 0

    async def chat_json(self, system, user, **options):
        if "catalog" not in user:
            self.started += 1
            self.usage.prompt_tokens += self.tokens
        await asyncio.sleep(self.seconds)
        return await super().chat_json(system, user, **options)


async def many(calls):
    """Несколько запросов к приложению одновременно, в одном цикле событий."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as http:
        return await asyncio.gather(*(call(http) for call in calls))


def test_experts_are_listed_with_names_from_dataset(client):
    app.state.dataset = {**LIVE, "experts": [
        {"expert_id": "clinician", "role": "Сомнолог-клиницист. Думает как врач.", "speech_style": "Спокойно."},
    ]}
    body = client.get("/api/experts").json()
    assert body["user_id"] == "u1"
    assert body["experts"] == [{"expert_id": "clinician", "name": "Сомнолог-клиницист",
                                "description": "Думает как врач.", "speech_style": "Спокойно."}]


def test_chat_returns_chosen_expert_and_all_three_voices(client, monkeypatch):
    body = ask(client, monkeypatch, FakeLLM(good_answer())).json()
    assert body["expert_id"] == "biohacker" and body["text"] == body["answers"]["biohacker"]["text"]
    assert set(body["answers"]) == set(EXPERTS) and len({v["text"] for v in body["answers"].values()}) == 3
    assert body["refused"] is False and body["degraded"] is False
    assert body["citations"] == [{"article_id": "a2", "section_id": "s1", "article_title": "Caffeine timing and sleep",
                                  "section_title": "Results", "year": None, "doi": None}]


def test_chat_passes_stored_and_live_history_to_the_model(client, monkeypatch):
    llm = FakeLLM(good_answer())
    ask(client, monkeypatch, llm, history=[
        {"role": "user", "text": "А чай считается?"}, {"role": "assistant", "text": "В чае тоже есть кофеин."},
    ])
    assert llm.payloads[0]["history"] == [
        "user: Пью кофе в шесть вечера.", "user: А чай считается?", "agent: В чае тоже есть кофеин.",
    ]


def test_chat_keeps_more_history_than_the_dataset_run(client, monkeypatch):
    llm = FakeLLM(good_answer())
    turns = [{"role": "user" if i % 2 == 0 else "assistant", "text": f"реплика {i}"} for i in range(10)]
    ask(client, monkeypatch, llm, history=turns)
    assert len(llm.payloads[0]["history"]) == config.CHAT_HISTORY_MESSAGES > pipeline.HISTORY_MESSAGES
    assert llm.payloads[0]["history"][-1] == "agent: реплика 9"


def test_refusal_is_an_expert_answer_not_an_error(client, monkeypatch):
    refusal = good_answer(forbidden="dosage", kind="refusal", texts={e: REFUSAL for e in EXPERTS})
    response = ask(client, monkeypatch, FakeLLM(refusal))
    assert response.status_code == 200
    assert response.json()["refused"] is True and response.json()["text"] == REFUSAL.strip()


def test_refusal_flag_belongs_to_the_chosen_expert(client, monkeypatch):
    # У биохакера в обеих попытках небезопасный текст: ему уходит запасной отказ, остальным — обычный ответ.
    texts = {e: f"{BODY} Подпись {e}." for e in EXPERTS} | {"biohacker": f"{BODY} Хватит 100 мг кофеина."}
    body = ask(client, monkeypatch, FakeLLM(good_answer(texts=texts))).json()
    assert body["refused"] is True and body["degraded"] is True
    assert body["answers"]["clinician"]["refused"] is False
    other = ask(client, monkeypatch, FakeLLM(good_answer(texts=texts)), expert_id="clinician").json()
    assert other["refused"] is False and other["degraded"] is False


def test_same_question_is_answered_from_cache_for_another_expert(client, monkeypatch):
    llm = FakeLLM(good_answer())
    first = ask(client, monkeypatch, llm).json()
    calls = len(llm.payloads)
    second = ask(client, monkeypatch, llm, expert_id="clinician").json()
    assert len(llm.payloads) == calls  # переключение эксперта не стоит токенов
    assert second["text"] == first["answers"]["clinician"]["text"]


def test_failed_model_gives_fallback_text_and_is_not_cached(client, monkeypatch):
    body = ask(client, monkeypatch, FakeLLM(fail=True)).json()
    assert body["degraded"] is True and body["text"]
    assert ask(client, monkeypatch, FakeLLM(good_answer())).json()["degraded"] is False


def test_chat_rewrites_for_substance_but_not_for_style(client, monkeypatch):
    short = good_answer(texts={e: f"В исследовании кофе мешал уснуть. Подпись {e}." for e in EXPERTS})
    llm = FakeLLM(short, good_answer())
    assert ask(client, monkeypatch, llm).json()["text"].startswith("В исследовании кофе мешал уснуть")
    assert len(llm.payloads) == 1  # замечание к длине не стоит человеку второго ожидания

    dose = good_answer(texts={e: f"{BODY} Хватит 100 мг кофеина. Подпись {e}." for e in EXPERTS})
    llm = FakeLLM(dose, good_answer())
    body = ask(client, monkeypatch, llm, text="А сколько кофеина можно?").json()
    assert len(llm.payloads) == 2 and "100 мг" not in body["text"]

    # Нейросеть решила «ответ», а написала отказ: это противоречие по существу, его нельзя показать как ответ.
    mismatch = good_answer(texts={e: f"Дозу не назову. {BODY} Подпись {e}." for e in EXPERTS})
    llm = FakeLLM(mismatch, good_answer())
    body = ask(client, monkeypatch, llm, text="А если очень хочется?").json()
    assert len(llm.payloads) == 2 and "не назову" not in body["text"]


def test_chat_stops_at_daily_budget_and_gets_more_after_run(client, monkeypatch):
    monkeypatch.setattr(config, "CHAT_TOKEN_ESTIMATE", 100)
    monkeypatch.setattr(config, "CHAT_TOKEN_BUDGET", 150)
    monkeypatch.setattr(config, "CHAT_TOKEN_BUDGET_AFTER_RUN", 1000)
    llm = FakeLLM(good_answer())
    llm.usage.prompt_tokens = 150  # столько «потратит» первый разговор
    assert ask(client, monkeypatch, llm).status_code == 200
    blocked = ask(client, monkeypatch, llm, text="Другой вопрос про кофе?")
    assert blocked.status_code == 429 and "Retry-After" in blocked.headers
    app.state.run_finished = True
    assert ask(client, monkeypatch, llm, text="Другой вопрос про кофе?").status_code == 200


def test_chat_leaves_room_for_what_the_run_already_spent(client, monkeypatch):
    app.state.run_finished = True
    chat.state_of(app).note_run(config.DAILY_TOKEN_LIMIT - config.DAILY_TOKEN_MARGIN - config.CHAT_TOKEN_ESTIMATE + 1)
    assert ask(client, monkeypatch, FakeLLM(good_answer())).status_code == 429


def test_parallel_questions_cannot_slip_past_the_budget(client, monkeypatch):
    monkeypatch.setattr(config, "CHAT_TOKEN_ESTIMATE", 100)
    monkeypatch.setattr(config, "CHAT_TOKEN_BUDGET", 250)  # хватает на два вопроса по 100
    monkeypatch.setattr(chat, "MAX_QUESTIONS_IN_WORK", 50)
    llm = SlowLLM(good_answer(), tokens=100)
    monkeypatch.setattr(pipeline, "LLM", lambda: llm)
    responses = asyncio.run(many(
        [lambda http, i=i: http.post("/api/chat", json={**QUESTION, "text": f"Вопрос {i} про кофе?"}) for i in range(8)]
    ))
    assert sorted(r.status_code for r in responses) == [200, 200] + [429] * 6
    assert llm.started == 2 and chat.state_of(app).reserved == 0


def test_queue_of_questions_is_bounded(client, monkeypatch):
    llm = SlowLLM(good_answer())
    monkeypatch.setattr(pipeline, "LLM", lambda: llm)
    responses = asyncio.run(many(
        [lambda http, i=i: http.post("/api/chat", json={**QUESTION, "text": f"Вопрос {i} про кофе?"}) for i in range(7)]
    ))
    assert [r.status_code for r in responses].count(200) == chat.MAX_QUESTIONS_IN_WORK
    assert [r.status_code for r in responses].count(429) == 7 - chat.MAX_QUESTIONS_IN_WORK


def test_same_question_asked_twice_at_once_is_computed_once(client, monkeypatch):
    llm = SlowLLM(good_answer())
    monkeypatch.setattr(pipeline, "LLM", lambda: llm)
    first, second = asyncio.run(many([
        lambda http: http.post("/api/chat", json=QUESTION),
        lambda http: http.post("/api/chat", json={**QUESTION, "expert_id": "clinician"}),
    ]))
    assert first.status_code == second.status_code == 200 and llm.started == 1
    assert second.json()["text"] == first.json()["answers"]["clinician"]["text"]


def test_chat_waits_while_a_run_is_in_progress(client, monkeypatch):
    class Unfinished:
        def done(self):
            return False

    app.state.runs[("code", "token")] = Unfinished()
    response = ask(client, monkeypatch, FakeLLM(good_answer()))
    assert response.status_code == 503 and response.headers["Retry-After"] == "60"
    app.state.runs.clear()


def test_run_interrupts_questions_already_in_work(client, monkeypatch):
    llm = SlowLLM(good_answer(), seconds=0.3)
    monkeypatch.setattr(pipeline, "LLM", lambda: llm)

    async def fake_fetch(token, dataset_code):
        return DATASET

    monkeypatch.setattr(hack_api, "fetch_dataset", fake_fetch)

    async def run_after_chats_started(http):
        await asyncio.sleep(0.05)
        return await http.post("/run", json={"case": 4, "dataset_code": "case4-test"},
                               headers={"Authorization": "Bearer run-token"})

    *chats, run = asyncio.run(many(
        [lambda http, i=i: http.post("/api/chat", json={**QUESTION, "text": f"Вопрос {i} про кофе?"}) for i in range(4)]
        + [run_after_chats_started]
    ))
    assert run.status_code == 200 and len(run.json()["requests"]) == 3
    assert [r.status_code for r in chats] == [503] * 4  # и те, что считались, и те, что ждали очереди
    assert chat.state_of(app).in_work == {} and chat.state_of(app).reserved == 0


def test_no_new_questions_in_the_window_before_the_scored_run(client, monkeypatch):
    monkeypatch.setattr(config, "CHAT_FREEZE_FROM", "2000-01-01T00:00:00+03:00")
    monkeypatch.setattr(config, "CHAT_FREEZE_UNTIL", "2999-01-01T00:00:00+03:00")
    llm = FakeLLM(good_answer())
    assert ask(client, monkeypatch, llm).status_code == 503 and not llm.payloads
    app.state.run_finished = True  # зачётный прогон прошёл — можно разговаривать
    assert ask(client, monkeypatch, llm).status_code == 200


def test_slow_answer_gives_504_and_the_retry_gets_the_finished_answer(client, monkeypatch):
    monkeypatch.setattr(config, "CHAT_TIMEOUT_SECONDS", 0.05)  # сервер ждёт не меньше секунды
    llm = SlowLLM(good_answer(), seconds=0.8)
    monkeypatch.setattr(pipeline, "LLM", lambda: llm)

    async def scenario(http):
        late = await http.post("/api/chat", json=QUESTION)
        await asyncio.sleep(1.0)  # вопрос досчитывается без клиента
        return late, await http.post("/api/chat", json=QUESTION)

    (late, retry), = asyncio.run(many([scenario]))
    assert late.status_code == 504 and retry.status_code == 200 and llm.started == 1


def test_unknown_expert_and_bad_input_are_rejected(client, monkeypatch):
    assert ask(client, monkeypatch, FakeLLM(good_answer()), expert_id="astrologer").status_code == 404
    assert ask(client, monkeypatch, FakeLLM(good_answer()), text="").status_code == 422
    assert ask(client, monkeypatch, FakeLLM(good_answer()), text="я" * 2001).status_code == 422
    assert ask(client, monkeypatch, FakeLLM(good_answer()), user_id="u" * 65).status_code == 422


def test_oversized_body_is_rejected_before_it_is_read(client):
    huge = b'{"expert_id": "biohacker", "text": "' + b"x" * (config.API_MAX_BODY_BYTES + 1) + b'"}'
    declared = client.post("/api/chat", content=huge, headers={"Content-Type": "application/json"})
    assert declared.status_code == 413

    def chunks():  # длина не объявлена: сервер должен считать сам
        for start in range(0, len(huge), 8192):
            yield huge[start:start + 8192]

    assert client.post("/api/chat", content=chunks(), headers={"Content-Type": "application/json"}).status_code == 413
    assert client.post("/run", content=huge[:100], headers={"Content-Type": "application/json"}).status_code != 413


def test_source_text_is_served_for_citation(client):
    body = client.get("/api/sources/a2/s1").json()
    assert body["text"] == "Caffeine before bedtime reduced sleep time." and body["article_title"] == "Caffeine timing and sleep"
    assert client.get("/api/sources/a2/s9").status_code == 404


def test_browser_on_another_address_is_allowed(client):
    response = client.options("/api/chat", headers={
        "Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type",
    })
    assert response.status_code == 200 and response.headers["access-control-allow-origin"] == "*"


def test_dev_dataset_is_loaded_when_no_run_happened(client, monkeypatch):
    app.state.dataset = None
    seen = {}

    async def fake_fetch(token, dataset_code):
        seen["code"] = dataset_code
        return LIVE

    monkeypatch.setattr(hack_api, "fetch_dataset", fake_fetch)
    assert len(client.get("/api/experts").json()["experts"]) == 3
    assert seen["code"] == config.DEV_DATASET_CODE


def test_slow_dev_dataset_load_does_not_replace_the_dataset_of_a_run(client, monkeypatch):
    app.state.dataset = None
    final = {**LIVE, "experts": LIVE["experts"][:1]}

    async def slow_fetch(token, dataset_code):
        await asyncio.sleep(0.1)
        return LIVE

    monkeypatch.setattr(hack_api, "fetch_dataset", slow_fetch)

    async def run_arrives(http):
        await asyncio.sleep(0.02)
        app.state.dataset = final  # так /run сохраняет полученный набор

    first, _ = asyncio.run(many([lambda http: http.get("/api/experts"), run_arrives]))
    assert len(first.json()["experts"]) == 1 and app.state.dataset is final


def test_missing_dataset_is_a_clear_503(client, monkeypatch):
    app.state.dataset = None

    async def failing_fetch(token, dataset_code):
        raise hack_api.HackApiError(401, "нет доступа")

    monkeypatch.setattr(hack_api, "fetch_dataset", failing_fetch)
    assert client.get("/api/experts").status_code == 503


def test_calls_of_different_llm_objects_share_one_pace(monkeypatch):
    monkeypatch.setattr(llm_module, "MIN_SECONDS_BETWEEN_CALLS", 0.05)
    monkeypatch.setattr(llm_module, "_next_start", 0.0)

    async def three_objects():
        started = asyncio.get_running_loop().time()
        await asyncio.gather(*(llm_module.LLM()._wait_turn() for _ in range(3)))
        return asyncio.get_running_loop().time() - started

    assert asyncio.run(three_objects()) >= 0.09  # три старта — два промежутка, хотя объекты разные


def test_retry_pause_never_outlasts_the_deadline(monkeypatch):
    slept = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(llm_module.asyncio, "sleep", fake_sleep)
    client = llm_module.LLM()
    client.deadline = llm_module.time.monotonic() + 3
    asyncio.run(client._pause(1, retry_after=60.0))
    assert slept and slept[0] <= 3
