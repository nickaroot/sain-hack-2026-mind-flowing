"""/run должен отдавать тело, которое проходит схему Case4Answers из спецификации."""

import json
from pathlib import Path

import jsonschema
import pytest
from fastapi.testclient import TestClient

from app import hack_api
from app.main import app

SPEC = json.loads((Path(__file__).parent.parent / "docs/api/openapi.json").read_text())
ANSWERS_SCHEMA = SPEC["components"]["schemas"]["Case4Answers"]

DATASET = {
    "meta": {"dataset_id": "case4-test"},
    "articles": [
        {"article_id": "a1", "sections": [{"section_id": "s1", "title": "t", "text": "x"}]}
    ],
    "requests": [
        {"record_id": f"r{i}", "user_id": "u1", "expert_id": expert, "group_id": "g1",
         "utterance": {"role": "user", "at": "2026-11-12T08:00:00+03:00", "text": "Как лучше засыпать?"}}
        for i, expert in enumerate(["clinician", "biohacker", "nutritionist"])
    ],
}


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


def test_health(client):
    assert client.get("/").status_code == 200


def test_run_answers_every_request_in_schema(client):
    response = client.post(
        "/run",
        json={"case": 4, "dataset_code": "case4-test"},
        headers={"Authorization": "Bearer run-token"},
    )
    assert response.status_code == 200
    body = response.json()
    jsonschema.validate(body, ANSWERS_SCHEMA)
    assert [a["record_id"] for a in body["requests"]] == ["r0", "r1", "r2"]


def test_run_forwards_token_from_request(client):
    client.post("/run", json={"case": 4, "dataset_code": "c"},
                headers={"Authorization": "Bearer run-token"})
    assert client.seen == {"token": "run-token", "dataset_code": "c"}


def test_run_rejects_missing_token_and_wrong_case(client):
    assert client.post("/run", json={"case": 4}).status_code == 401
    assert client.post("/run", json={"case": 1},
                       headers={"Authorization": "Bearer t"}).status_code == 400
