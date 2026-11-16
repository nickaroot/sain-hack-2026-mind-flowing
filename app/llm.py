"""Вызов нейросети через прокси сервиса хакатона (API как у OpenAI).

Адрес, ключ и модель берутся только из окружения. Повторы делаем сами и с
паузой: каждый повтор тоже идёт в суточный лимит токенов.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field

import httpx

from app import config

log = logging.getLogger("app.llm")

RETRY_PAUSES_SECONDS = (5, 10, 20)
REQUEST_TIMEOUT_SECONDS = 180.0
MAX_PARALLEL_CALLS = 4  # лимит прокси — 30 запросов в минуту


class LLMError(RuntimeError):
    pass


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    failures: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, usage: dict) -> None:
        self.calls += 1
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        self.reasoning_tokens += (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)


@dataclass
class LLM:
    usage: Usage = field(default_factory=Usage)
    _gate: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(MAX_PARALLEL_CALLS))

    async def chat_json(self, system: str, user: str, *, max_tokens: int, effort: str = "low") -> dict:
        """Один вызов с ответом в виде JSON-объекта. Бросает LLMError, если ответа получить не удалось."""
        body = {
            "model": config.LLM_MODEL,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_completion_tokens": max_tokens,
            "reasoning_effort": effort,
            "response_format": {"type": "json_object"},
        }
        if not config.LLM_BASE_URL or not config.LLM_API_KEY:
            self.usage.failures += 1
            raise LLMError("нейросеть не настроена: нет LLM_BASE_URL или LLM_API_KEY")
        headers = {"Authorization": f"Bearer {config.LLM_API_KEY}"}
        last_error = "нет попыток"
        async with self._gate:
            for pause in (0, *RETRY_PAUSES_SECONDS):
                if pause:
                    await asyncio.sleep(pause)
                try:
                    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
                        response = await client.post(
                            f"{config.LLM_BASE_URL}/chat/completions", headers=headers, json=body
                        )
                except httpx.HTTPError as error:
                    last_error = type(error).__name__
                    continue
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = f"HTTP {response.status_code}"
                    continue
                if response.status_code != 200:
                    raise LLMError(f"HTTP {response.status_code}")
                payload = response.json()
                self.usage.add(payload.get("usage") or {})
                choice = payload["choices"][0]
                if choice.get("finish_reason") == "length":
                    last_error = "ответ обрезан по max_tokens"
                    break
                try:
                    parsed = json.loads(choice["message"]["content"])
                except (TypeError, ValueError):
                    last_error = "ответ не разбирается как JSON"
                    continue
                if isinstance(parsed, dict):
                    return parsed
                last_error = "ответ не JSON-объект"
        self.usage.failures += 1
        log.warning("вызов нейросети не удался: %s", last_error)
        raise LLMError(last_error)
