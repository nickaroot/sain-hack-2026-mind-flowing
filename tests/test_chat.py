"""Методы для клиента: эксперты, разговор и источник. Разговор не должен мешать прогону набора."""

import pytest
from fastapi.testclient import TestClient

from app import config, hack_api, pipeline
from app.main import app
from tests.test_run import BODY, DATASET, EXPERTS, REFUSAL, FakeLLM, good_answer

LIVE = {**DATASET, "users": [{"user_id": "u1"}], "chat_history": [
    {"user_id": "u1", "role": "user", "at": "2026-11-10T22:00:00+03:00", "text": "Пью кофе в шесть вечера."},
]}


@pytest.fixture
def client(monkeypatch):
    app.state.dataset = LIVE
    app.state.runs.clear()
    app.state.run_finished = False
    app.state.chat = None
    yield TestClient(app)
    app.state.dataset = None
    app.state.chat = None
    app.state.run_finished = False


def ask(client, monkeypatch, llm, **body):
    monkeypatch.setattr(pipeline, "LLM", lambda: llm)
    return client.post("/api/chat", json={"expert_id": "biohacker", "text": "Мешает ли кофе вечером уснуть?", **body})


def test_experts_are_listed_with_names_from_dataset(client):
    app.state.dataset = {**LIVE, "experts": [
        {"expert_id": "clinician", "role": "Сомнолог-клиницист. Думает как врач.", "speech_style": "Спокойно."},
    ]}
    body = client.get("/api/experts").json()
    assert body["user_id"] == "u1"
    assert body["experts"] == [{"expert_id": "clinician", "name": "Сомнолог-клиницист",
                                "description": "Думает как врач.", "speech_style": "Спокойно."}]


def test_chat_returns_chosen_expert_and_all_three_voices(client, monkeypatch):
    llm = FakeLLM(good_answer())
    body = ask(client, monkeypatch, llm).json()
    assert body["expert_id"] == "biohacker" and body["text"] == body["answers"]["biohacker"]
    assert set(body["answers"]) == set(EXPERTS) and len(set(body["answers"].values())) == 3
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


def test_refusal_is_an_expert_answer_not_an_error(client, monkeypatch):
    refusal = good_answer(forbidden="dosage", kind="refusal", texts={e: REFUSAL for e in EXPERTS})
    response = ask(client, monkeypatch, FakeLLM(refusal))
    assert response.status_code == 200
    assert response.json()["refused"] is True and response.json()["text"] == REFUSAL.strip()


def test_same_question_is_answered_from_cache_for_another_expert(client, monkeypatch):
    llm = FakeLLM(good_answer())
    first = ask(client, monkeypatch, llm).json()
    calls = len(llm.payloads)
    second = ask(client, monkeypatch, llm, expert_id="clinician").json()
    assert len(llm.payloads) == calls  # переключение эксперта не стоит токенов
    assert second["text"] == first["answers"]["clinician"]


def test_failed_model_gives_fallback_text_and_is_not_cached(client, monkeypatch):
    body = ask(client, monkeypatch, FakeLLM(fail=True)).json()
    assert body["degraded"] is True and body["text"]
    assert ask(client, monkeypatch, FakeLLM(good_answer())).json()["degraded"] is False


def test_chat_stops_at_daily_budget_and_gets_more_after_run(client, monkeypatch):
    monkeypatch.setattr(config, "CHAT_TOKEN_BUDGET", 100)
    monkeypatch.setattr(config, "CHAT_TOKEN_BUDGET_AFTER_RUN", 1000)
    llm = FakeLLM(good_answer())
    llm.usage.prompt_tokens = 150  # столько «потратит» первый разговор
    assert ask(client, monkeypatch, llm).status_code == 200
    blocked = ask(client, monkeypatch, llm, text="Другой вопрос про кофе?")
    assert blocked.status_code == 429 and "Retry-After" in blocked.headers
    app.state.run_finished = True
    assert ask(client, monkeypatch, llm, text="Другой вопрос про кофе?").status_code == 200


def test_chat_waits_while_a_run_is_in_progress(client, monkeypatch):
    class Unfinished:
        def done(self):
            return False

    app.state.runs[("code", "token")] = Unfinished()
    response = ask(client, monkeypatch, FakeLLM(good_answer()))
    assert response.status_code == 503 and response.headers["Retry-After"] == "60"
    app.state.runs.clear()


def test_unknown_expert_and_bad_input_are_rejected(client, monkeypatch):
    assert ask(client, monkeypatch, FakeLLM(good_answer()), expert_id="astrologer").status_code == 404
    assert ask(client, monkeypatch, FakeLLM(good_answer()), text="").status_code == 422
    assert ask(client, monkeypatch, FakeLLM(good_answer()), text="я" * 2001).status_code == 422


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


def test_missing_dataset_is_a_clear_503(client, monkeypatch):
    app.state.dataset = None

    async def failing_fetch(token, dataset_code):
        raise hack_api.HackApiError(401, "нет доступа")

    monkeypatch.setattr(hack_api, "fetch_dataset", failing_fetch)
    assert client.get("/api/experts").status_code == 503


def test_chat_does_not_rewrite_for_style_but_does_for_unsafe_text(client, monkeypatch):
    short = good_answer(texts={e: f"В исследовании кофе мешал уснуть. Подпись {e}." for e in EXPERTS})
    llm = FakeLLM(short, good_answer())
    assert ask(client, monkeypatch, llm).json()["text"].startswith("В исследовании кофе мешал уснуть")
    assert len(llm.payloads) == 1  # замечание к длине не стоит человеку второго ожидания

    dose = good_answer(texts={e: f"{BODY} Хватит 100 мг кофеина. Подпись {e}." for e in EXPERTS})
    llm = FakeLLM(dose, good_answer())
    body = ask(client, monkeypatch, llm, text="А сколько кофеина можно?").json()
    assert len(llm.payloads) == 2 and "100 мг" not in body["text"]
