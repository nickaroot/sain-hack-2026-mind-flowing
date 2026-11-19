"""Получение набора у сервиса хакатона: без набора прогон даёт ноль, поэтому сбои повторяются."""

import asyncio

import httpx
import pytest

from app import hack_api
from app.hack_api import HackApiError


@pytest.fixture
def service(monkeypatch):
    """Подменяет сервис хакатона: отвечает по очереди тем, что положено в .replies."""

    class Service:
        def __init__(self):
            self.replies: list = []
            self.requests: list[httpx.Request] = []

    seen = Service()

    def respond(request: httpx.Request) -> httpx.Response:
        seen.requests.append(request)
        reply = seen.replies.pop(0) if len(seen.replies) > 1 else seen.replies[0]
        if isinstance(reply, Exception):
            raise reply
        status, payload = reply
        return httpx.Response(status, json=payload)

    async def no_pause(seconds):
        return None

    real_client = httpx.AsyncClient
    monkeypatch.setattr(hack_api.httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(respond)))
    monkeypatch.setattr(hack_api.asyncio, "sleep", no_pause)
    monkeypatch.setattr(hack_api.config, "HACK_API_URL", "http://hack.test")
    return seen


def test_dataset_is_requested_with_the_token_of_the_incoming_run(service):
    service.replies = [(200, {"dataset": {"requests": []}})]
    assert asyncio.run(hack_api.fetch_dataset("run-token", "code-1")) == {"requests": []}
    sent = service.requests[0]
    assert sent.url.path == "/case/4/getData" and sent.url.params["dataset_code"] == "code-1"
    assert sent.headers["authorization"] == "Bearer run-token"
    asyncio.run(hack_api.fetch_dataset("run-token", None))
    assert "dataset_code" not in service.requests[1].url.params


def test_server_errors_and_network_failures_are_retried(service):
    service.replies = [(503, {"error": "busy"}), httpx.ConnectError("нет сети"), (200, {"dataset": {"requests": []}})]
    assert asyncio.run(hack_api.fetch_dataset("t", None)) == {"requests": []}
    assert len(service.requests) == 3


def test_refusal_of_the_service_is_not_retried(service):
    service.replies = [(403, {"error": "wrong token"})]
    with pytest.raises(HackApiError) as error:
        asyncio.run(hack_api.fetch_dataset("t", "code"))
    assert error.value.status_code == 403 and len(service.requests) == 1


def test_persistent_failure_reports_the_last_error(service):
    service.replies = [(500, {"error": "oops"})]
    with pytest.raises(HackApiError) as error:
        asyncio.run(hack_api.fetch_dataset("t", None))
    assert error.value.status_code == 500 and len(service.requests) == 3
    service.replies = [httpx.ReadTimeout("долго")]
    with pytest.raises(HackApiError) as error:
        asyncio.run(hack_api.fetch_dataset("t", None))
    assert error.value.status_code == 0 and "ReadTimeout" in str(error.value)
