"""Клиент сервиса хакатона: только то, что нужно решению в рантайме."""

import asyncio

import httpx

from app import config
from app.contracts import Dataset


class HackApiError(RuntimeError):
    def __init__(self, status_code: int, body: str):
        super().__init__(f"hack-api ответил {status_code}: {body[:300]}")
        self.status_code = status_code


async def fetch_dataset(token: str, dataset_code: str | None) -> Dataset:
    """GET /case/4/getData. Токен берётся из входящего /run, а не из окружения."""
    params = {"dataset_code": dataset_code} if dataset_code else {}
    last = HackApiError(0, "нет попыток")
    # Без набора весь прогон даёт ноль, поэтому сетевой сбой и ответ 5xx повторяем; 4xx — это отказ, его не повторить.
    for pause in (0, 3, 10):
        if pause:
            await asyncio.sleep(pause)
        try:
            async with httpx.AsyncClient(timeout=config.HACK_API_TIMEOUT_SECONDS) as client:
                response = await client.get(
                    f"{config.HACK_API_URL}/case/{config.CASE_ID}/getData",
                    params=params,
                    headers={"Authorization": f"Bearer {token}"},
                )
        except httpx.HTTPError as error:
            last = HackApiError(0, type(error).__name__)
            continue
        if response.status_code == 200:
            return response.json()["dataset"]
        last = HackApiError(response.status_code, response.text)
        if response.status_code < 500:
            break
    raise last
