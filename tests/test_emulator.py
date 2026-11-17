"""Эмулятор стенда должен вести себя как сервисы организаторов там, где это задокументировано."""

import json

import pytest
from fastapi.testclient import TestClient

from scripts import emulator
from tests.test_run import DATASET

TEAM = {"Authorization": f"Bearer {emulator.TEAM_TOKEN}"}
RUN = {"Authorization": f"Bearer {emulator.RUN_TOKEN}"}
LLM = {"Authorization": f"Bearer {emulator.LLM_KEY}"}
REFERENCE = {"requests": [{"record_id": r["record_id"], "request_type": "normal"} for r in DATASET["requests"]]}


@pytest.fixture
def make_client(tmp_path, monkeypatch):
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps({"dataset": DATASET, "reference": REFERENCE}), encoding="utf-8")

    async def reply(self, system, user, **_):
        return {"echo": "ответ"}

    monkeypatch.setattr(emulator.CodexLLM, "chat_json", reply)
    monkeypatch.setattr(emulator.CodexLLM, "__init__", lambda self, *a, **k: None)

    def make(*extra: str) -> TestClient:
        return TestClient(emulator.create_app(emulator.parse_args(["--dataset", f"code={path}", *extra])))

    return make


def chat(client: TestClient, **options):
    body = {"model": emulator.MODEL, "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]}
    return client.post("/v1/chat/completions", json={**body, **options}, headers=LLM)


def test_run_token_gets_dataset_without_reference(make_client):
    client = make_client()
    assert client.get("/case/4/getData", params={"dataset_code": "code"}, headers=TEAM).json()["reference"]
    hidden = client.get("/case/4/getData", params={"dataset_code": "code"}, headers=RUN).json()
    assert hidden["reference"] is None and hidden["dataset"]["meta"]["phase"] == "online"
    assert client.get("/case/4/getData", params={"dataset_code": "чужой"}, headers=TEAM).status_code == 403
    assert client.get("/case/4/getData", params={"dataset_code": "code"}).status_code == 401


def test_validate_checks_schema_coverage_and_citations(make_client):
    client = make_client()
    answer = {"record_id": "r0", "text": "т", "refused": False, "citations": [{"article_id": "a1", "section_id": "s1"}]}
    full = [{**answer, "record_id": r["record_id"]} for r in DATASET["requests"]]
    assert client.post("/case/4/validate", json={"requests": full}, headers=TEAM).json()["valid"]
    assert not client.post("/case/4/validate", json={"requests": full[:1]}, headers=TEAM).json()["valid"]
    broken = [{**a, "citations": [{"article_id": "a1", "section_id": "нет"}]} for a in full]
    assert not client.post("/case/4/validate", json={"requests": broken}, headers=TEAM).json()["valid"]
    assert client.post("/case/4/validate", json={"requests": full}, headers=RUN).status_code == 403


def test_proxy_enforces_requests_per_minute_and_daily_tokens(make_client):
    client = make_client("--rpm", "2")
    assert [chat(client).status_code for _ in range(3)] == [200, 200, 429]
    spent = make_client("--spent", "300000")
    limited = chat(spent)
    assert limited.status_code == 429 and int(limited.headers["retry-after"]) > 120


def test_proxy_rejects_parameters_of_another_model_and_reports_usage(make_client):
    client = make_client("--reject", "reasoning_effort")
    assert chat(client, reasoning_effort="low").status_code == 400
    reply = chat(client).json()
    assert json.loads(reply["choices"][0]["message"]["content"]) == {"echo": "ответ"}
    assert reply["usage"]["total_tokens"] == reply["usage"]["prompt_tokens"] + reply["usage"]["completion_tokens"]
    assert chat(client, max_completion_tokens=64).json()["choices"] == []  # лимит съеден рассуждением


def test_token_estimate_matches_proxy_measurements():
    # Замеры настоящего прокси 17.11: 1800 символов русского текста — около 473 токенов.
    text = ("Вечерний свет связан с более поздним временем сна. " * 40)[:1800]
    assert 430 < emulator.estimate_tokens(text) < 520
