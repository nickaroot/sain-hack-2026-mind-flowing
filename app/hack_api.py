"""Клиент сервиса хакатона: только то, что нужно решению в рантайме."""

import httpx

from app import config


class HackApiError(RuntimeError):
    def __init__(self, status_code: int, body: str):
        super().__init__(f"hack-api ответил {status_code}: {body[:300]}")
        self.status_code = status_code


async def fetch_dataset(token: str, dataset_code: str | None) -> dict:
    """GET /case/4/getData. Токен берётся из входящего /run, а не из окружения."""
    params = {"dataset_code": dataset_code} if dataset_code else {}
    async with httpx.AsyncClient(timeout=config.HACK_API_TIMEOUT_SECONDS) as client:
        response = await client.get(
            f"{config.HACK_API_URL}/case/{config.CASE_ID}/getData",
            params=params,
            headers={"Authorization": f"Bearer {token}"},
        )
    if response.status_code != 200:
        raise HackApiError(response.status_code, response.text)
    return response.json()["dataset"]
