"""Вызов нейросети через прокси сервиса хакатона (API как у OpenAI).

Адрес, ключ и модель берутся только из окружения. Повторы делаем сами и с
паузой: каждый повтор тоже идёт в суточный лимит токенов.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

import httpx

from app import config

log = logging.getLogger("app.llm")

RETRY_PAUSES_SECONDS = (5, 10, 20)
REQUEST_TIMEOUT_SECONDS = 120.0
MAX_PARALLEL_CALLS = 4
MIN_SECONDS_BETWEEN_CALLS = 2.2  # лимит прокси — 30 запросов в минуту, повторы тоже считаются
MAX_RETRY_AFTER_SECONDS = 60.0


class LLMError(RuntimeError):
    pass


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    cached_tokens: int = 0
    failures: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def add(self, usage: dict) -> None:
        self.calls += 1
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        self.reasoning_tokens += (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        self.cached_tokens += (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)


# Наборы необязательных параметров запроса от полного к самому простому.
_OPTION_VARIANTS = (
    ("max_completion_tokens", "reasoning_effort", "response_format"),
    ("max_completion_tokens", "response_format"),
    ("max_completion_tokens",),
    ("max_tokens",),
)


def _request_options(variant: int, max_tokens: int, effort: str) -> dict:
    values = {
        "max_completion_tokens": max_tokens,
        "max_tokens": max_tokens + 1000,  # без управления рассуждением модели нужен запас на скрытые токены
        "reasoning_effort": effort,
        "response_format": {"type": "json_object"},
    }
    return {name: values[name] for name in _OPTION_VARIANTS[variant]}


def _parse_object(content: str):
    """JSON из ответа модели. Без response_format модель может обернуть его в ```json или добавить фразу."""
    try:
        return json.loads(content)
    except ValueError:
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(content[start : end + 1])


def _retry_after(response: httpx.Response) -> float:
    """Пауза, которую просит прокси в заголовке Retry-After, в разумных пределах."""
    try:
        return min(max(float(response.headers.get("retry-after", 0)), 0.0), MAX_RETRY_AFTER_SECONDS)
    except ValueError:
        return 0.0


@dataclass
class LLM:
    usage: Usage = field(default_factory=Usage)
    deadline: float | None = None  # момент по time.monotonic(), после которого вызовы не начинаются
    _variant: int = 0  # номер набора необязательных параметров, который принимает модель
    _pace: asyncio.Lock = field(default_factory=asyncio.Lock)
    _next_start: float = 0.0

    async def _wait_turn(self) -> None:
        """Запросы стартуют не чаще одного в MIN_SECONDS_BETWEEN_CALLS, сколько бы их ни шло параллельно."""
        async with self._pace:
            delay = self._next_start - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self._next_start = time.monotonic() + MIN_SECONDS_BETWEEN_CALLS
    _gate: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(MAX_PARALLEL_CALLS))

    async def chat_json(self, system: str, user: str, *, max_tokens: int, effort: str = "low") -> dict:
        """Один вызов с ответом в виде JSON-объекта. Бросает LLMError, если ответа получить не удалось."""
        base = {
            "model": config.LLM_MODEL,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if not config.LLM_BASE_URL or not config.LLM_API_KEY:
            self.usage.failures += 1
            raise LLMError("нейросеть не настроена: нет LLM_BASE_URL или LLM_API_KEY")
        headers = {"Authorization": f"Bearer {config.LLM_API_KEY}"}
        last_error = "нет попыток"
        async with self._gate:
            for pause in (0, *RETRY_PAUSES_SECONDS, *(0,) * (len(_OPTION_VARIANTS) - 1)):
                if pause:
                    await asyncio.sleep(pause)
                timeout = REQUEST_TIMEOUT_SECONDS
                if self.deadline is not None:
                    timeout = min(timeout, self.deadline - time.monotonic())
                    if timeout < 5:
                        last_error = "вышло время, отведённое на прогон"
                        break
                await self._wait_turn()
                variant = self._variant
                body = {**base, **_request_options(variant, max_tokens, effort)}
                try:
                    async with httpx.AsyncClient(timeout=timeout) as client:
                        response = await client.post(
                            f"{config.LLM_BASE_URL}/chat/completions", headers=headers, json=body
                        )
                except httpx.HTTPError as error:
                    last_error = type(error).__name__
                    continue
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = f"HTTP {response.status_code}"
                    await asyncio.sleep(_retry_after(response))
                    continue
                if response.status_code == 400 and variant < len(_OPTION_VARIANTS) - 1:
                    # Организаторы могут сменить модель через LLM_MODEL, а другая модель может не знать
                    # наших необязательных параметров. Тогда пробуем более простой набор и запоминаем его.
                    if self._variant == variant:  # параллельные вызовы не должны перескочить через набор
                        self._variant = variant + 1
                        log.warning("нейросеть ответила 400: перехожу к набору параметров %d", self._variant)
                    last_error = "HTTP 400, упрощаем параметры запроса"
                    continue
                if response.status_code != 200:
                    last_error = f"HTTP {response.status_code}"
                    break
                try:
                    payload = response.json()
                    self.usage.add(payload.get("usage") or {})
                    choice = payload["choices"][0]
                    if choice.get("finish_reason") == "length":
                        last_error = "ответ обрезан по max_tokens"
                        break
                    parsed = _parse_object(choice["message"]["content"])
                except (TypeError, ValueError, KeyError, IndexError, AttributeError):
                    # прокси ответил 200, но не тем, что обещает протокол: это сбой одного вызова, а не прогона
                    last_error = "ответ не разбирается как JSON"
                    continue
                if isinstance(parsed, dict):
                    return parsed
                last_error = "ответ не JSON-объект"
        self.usage.failures += 1
        log.warning("вызов нейросети не удался: %s", last_error)
        raise LLMError(last_error)
