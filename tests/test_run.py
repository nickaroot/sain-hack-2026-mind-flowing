"""/run должен отдавать тело, которое проходит схему Case4Answers, в том числе когда нейросеть недоступна."""

import json
from pathlib import Path

import jsonschema
import pytest
from fastapi.testclient import TestClient

from app import hack_api, pipeline
from app.llm import LLMError, Usage
from app.main import app
from app.retrieval import Index

SPEC = json.loads((Path(__file__).parent.parent / "docs/api/openapi.json").read_text())
ANSWERS_SCHEMA = SPEC["components"]["schemas"]["Case4Answers"]
EXPERTS = ["clinician", "biohacker", "nutritionist"]

DATASET = {
    "meta": {"dataset_id": "case4-test"},
    "chat_history": [],
    "experts": [{"expert_id": e, "role": "роль", "speech_style": "стиль"} for e in EXPERTS],
    "articles": [
        {"article_id": "a1", "title": "Evening light and melatonin",
         "sections": [{"section_id": "s1", "title": "Results", "text": "Evening light exposure suppressed melatonin."}]},
        {"article_id": "a2", "title": "Caffeine timing and sleep",
         "sections": [{"section_id": "s1", "title": "Results", "text": "Caffeine before bedtime reduced sleep time."}]},
    ],
    "requests": [
        {"record_id": f"r{i}", "user_id": "u1", "expert_id": expert, "group_id": "g1",
         "utterance": {"role": "user", "at": "2026-11-12T08:00:00+03:00", "text": "Мешает ли кофе вечером уснуть?"}}
        for i, expert in enumerate(EXPERTS)
    ],
}


class FakeLLM:
    """Отвечает как нейросеть: сначала поисковые слова, затем ответ на тройку."""

    def __init__(self, answer: dict | None = None, fail: bool = False):
        self.usage = Usage()
        self.answer = answer
        self.fail = fail

    async def chat_json(self, system, user, **_):
        if self.fail:
            raise LLMError("недоступна")
        if isinstance(json.loads(user), list):
            return {"0": "caffeine bedtime sleep time"}
        return self.answer


def good_answer(**overrides) -> dict:
    base = {"forbidden": "none", "kind": "answer", "source": 1, "quote": "Caffeine before bedtime",
            "texts": {e: f"Текст эксперта {e}." for e in EXPERTS}}
    return {**base, **overrides}


@pytest.fixture
def client(monkeypatch):
    seen = {}

    async def fake_fetch(token, dataset_code):
        seen["token"], seen["dataset_code"] = token, dataset_code
        return DATASET

    monkeypatch.setattr(hack_api, "fetch_dataset", fake_fetch)
    test_client = TestClient(app)
    test_client.seen = seen
    return test_client


def run(client, monkeypatch, llm):
    monkeypatch.setattr(pipeline, "LLM", lambda: llm)
    response = client.post("/run", json={"case": 4, "dataset_code": "case4-test"},
                           headers={"Authorization": "Bearer run-token"})
    assert response.status_code == 200
    body = response.json()
    jsonschema.validate(body, ANSWERS_SCHEMA)
    assert [a["record_id"] for a in body["requests"]] == ["r0", "r1", "r2"]
    return body["requests"]


def test_health(client):
    assert client.get("/").status_code == 200
    assert client.get("/health").json()["status"] == "ok"


def test_answers_share_citation_found_by_search(client, monkeypatch):
    answers = run(client, monkeypatch, FakeLLM(good_answer()))
    assert all(a["citations"] == [{"article_id": "a2", "section_id": "s1"}] for a in answers)
    assert not any(a["refused"] for a in answers)


def test_refusal_flag_follows_forbidden_request(client, monkeypatch):
    answers = run(client, monkeypatch, FakeLLM(good_answer(forbidden="dosage", kind="refusal")))
    assert all(a["refused"] for a in answers)


def test_text_with_dose_is_replaced_by_fallback(client, monkeypatch):
    texts = {e: "Обычный текст." for e in EXPERTS} | {"biohacker": "Прими 3 мг мелатонина."}
    answers = run(client, monkeypatch, FakeLLM(good_answer(texts=texts)))
    assert "мг" not in answers[1]["text"] and answers[1]["refused"]
    assert answers[1]["citations"] == answers[0]["citations"]


def test_every_record_answered_when_llm_is_down(client, monkeypatch):
    answers = run(client, monkeypatch, FakeLLM(fail=True))
    assert all(a["refused"] and a["citations"] for a in answers)


def test_run_forwards_token_from_request(client, monkeypatch):
    run(client, monkeypatch, FakeLLM(good_answer()))
    assert client.seen == {"token": "run-token", "dataset_code": "case4-test"}


def test_run_rejects_missing_token_and_wrong_case(client):
    assert client.post("/run", json={"case": 4}).status_code == 401
    assert client.post("/run", json={"case": 1}, headers={"Authorization": "Bearer t"}).status_code == 400


def test_search_ranks_matching_article_first():
    hits = Index(DATASET["articles"]).search("melatonin suppression evening light")
    assert hits[0].section.article_id == "a1"


@pytest.mark.parametrize("text,safe", [
    ("Последняя чашка кофе за шесть часов до сна.", True),
    ("Три миллиграмма на ночь.", True),  # числа словами фильтр не ловит: это задача инструкции
    ("По 3 мг за час до сна.", False),
    ("20 капель в воде.", False),
    ("", False),
])
def test_dose_filter(text, safe):
    assert pipeline.is_safe_text(text) is safe
