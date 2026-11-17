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
LIMIT_PATIENCE_SECONDS = 150.0  # сколько терпим непрерывные 429, прежде чем считать суточный лимит исчерпанным


# Лимит прокси — на команду, а не на объект: прогон набора и разговоры в клиенте создают разные LLM,
# но стартовать их вызовы должны в одной очереди. Момент следующего разрешённого старта общий для процесса.
_next_start = 0.0


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
    """Сколько секунд прокси просит подождать (заголовок Retry-After); 0, если не сказал."""
    try:
        return max(float(response.headers.get("retry-after", 0)), 0.0)
    except ValueError:
        return 0.0


@dataclass
class LLM:
    usage: Usage = field(default_factory=Usage)
    deadline: float | None = None  # момент по time.monotonic(), после которого вызовы не начинаются
    exhausted: bool = False  # суточный лимит кончился: дальнейшие вызовы бессмысленны и только тратят время
    _limited_since: float | None = None  # когда началась непрерывная серия ответов 429
    _variant: int = 0  # номер набора необязательных параметров, который принимает модель

    def _note_limit(self, retry_after: float) -> None:
        """Отличает минутный лимит (проходит сам) от суточного (ждать бесполезно).

        Минутный лимит снимается за минуту, а запросы у нас идут не чаще одного в 2,2 секунды. Поэтому
        просьба подождать дольше двух минут или непрерывные отказы дольше LIMIT_PATIENCE_SECONDS
        означают, что кончились токены за сутки.
        """
        now = time.monotonic()
        if self._limited_since is None:
            self._limited_since = now
        if retry_after > 120 or now - self._limited_since > LIMIT_PATIENCE_SECONDS:
            if not self.exhausted:
                log.error("суточный лимит нейросети исчерпан: оставшиеся записи получат запасные ответы")
            self.exhausted = True

    async def _pause(self, tries: int, retry_after: float = 0.0) -> None:
        if tries <= len(RETRY_PAUSES_SECONDS) and not self.exhausted:
            pause = max(RETRY_PAUSES_SECONDS[tries - 1], min(retry_after, MAX_RETRY_AFTER_SECONDS))
            if self.deadline is not None:  # ждать дольше срока незачем: после него вызов всё равно не начнётся
                pause = min(pause, max(0.0, self.deadline - time.monotonic()))
            await asyncio.sleep(pause)

    async def _wait_turn(self) -> None:
        """Запросы стартуют не чаще одного в MIN_SECONDS_BETWEEN_CALLS — все, от любого экземпляра LLM.

        Место в очереди занимается без await между чтением и записью, поэтому замок не нужен.
        """
        global _next_start
        now = time.monotonic()
        start = max(now, _next_start)
        _next_start = start + MIN_SECONDS_BETWEEN_CALLS
        if start > now:
            await asyncio.sleep(start - now)
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
            tries = 0
            while tries <= len(RETRY_PAUSES_SECONDS):
                if self.exhausted:
                    last_error = "суточный лимит нейросети исчерпан"
                    break
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
                    tries += 1
                    await self._pause(tries)
                    continue
                if response.status_code == 400 and variant < len(_OPTION_VARIANTS) - 1:
                    # Организаторы могут сменить модель через LLM_MODEL, а другая модель может не знать
                    # наших необязательных параметров. Тогда пробуем более простой набор и запоминаем его.
                    # Попытку это не тратит.
                    if self._variant == variant:  # параллельные вызовы не должны перескочить через набор
                        self._variant = variant + 1
                        log.warning("нейросеть ответила 400: перехожу к набору параметров %d", self._variant)
                    last_error = "HTTP 400, упрощаем параметры запроса"
                    continue
                if response.status_code == 429:
                    last_error = "HTTP 429"
                    self._note_limit(_retry_after(response))
                    tries += 1
                    await self._pause(tries, _retry_after(response))
                    continue
                if response.status_code >= 500:
                    last_error = f"HTTP {response.status_code}"
                    tries += 1
                    await self._pause(tries, _retry_after(response))
                    continue
                if response.status_code != 200:
                    last_error = f"HTTP {response.status_code}"
                    break
                self._limited_since = None  # прокси отвечает: лимит не исчерпан
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
                    tries += 1
                    continue
                if isinstance(parsed, dict):
                    return parsed
                last_error = "ответ не JSON-объект"
                tries += 1
        self.usage.failures += 1
        log.warning("вызов нейросети не удался: %s", last_error)
        raise LLMError(last_error)
