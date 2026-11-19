"""Эмулятор внешних сервисов стенда: сервис хакатона и прокси нейросети в одном процессе.

Зачем: прогнать решение целиком так, как его запустит стенд, не тратя суточный лимит прокси
организаторов, и проверить поведение в условиях, которые на настоящем прокси не воспроизвести
по заказу: исчерпанный лимит, 429, сбои, смена модели.

Что повторено по материалам организаторов и замерам:
- сервис хакатона: getData по коду набора (токен команды получает эталон, токен прогона — нет),
  validate (схема Case4Answers и существование ссылок), коды ошибок 400/401/403;
- прокси: API как у OpenAI, 30 запросов в минуту (скользящее окно) и суточный лимит токенов с
  ответом 429 и Retry-After, ответ 200 без choices, когда лимит max_completion_tokens съеден
  рассуждением (так ведёт себя настоящий прокси), поле usage с кешированными и скрытыми токенами;
- голос на том же ключе (доступен с 18.11): распознавание gpt-transcribe и синтез tts-1, по 10 запросов в минуту
  на каждое и в счёт общих 30, свой суточный бюджет в долларах с ответом 402, файл до 5 МБ; синтез возвращает
  заглушку вместо mp3, а распознавание такой заглушки — исходный текст, так что цепочку можно проверить насквозь;
- счёт токенов: коэффициенты измерены на настоящем прокси 17.11 (кириллица 3,41 символа на токен,
  остальное 4,47), ошибка на проверочном тексте около 6%.

Чего повторить нельзя: отвечает та же модель gpt-6-luna, но через Codex CLI по подписке
разработчика — без температуры и с описаниями инструментов Codex в запросе. Скрытые токены
рассуждения не измеряются, а берутся средними по замерам.

Запуск: python -m scripts.emulator --dataset case4-dev-v2=data/case4-dev-v2.json --cache emu-dev
"""

import argparse
import json
import random
import re
import time
from collections import deque
from email.parser import BytesParser
from email.policy import HTTP
from pathlib import Path

import jsonschema
import uvicorn
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response

from app.llm import LLMError
from scripts.offline_llm import CodexLLM

TEAM_TOKEN, RUN_TOKEN, LLM_KEY = "emu-team-token", "emu-run-token", "sk-bf-emulator"
MODEL = "gpt-6-luna"
CYRILLIC_CHARS_PER_TOKEN, OTHER_CHARS_PER_TOKEN, MESSAGE_OVERHEAD_TOKENS = 3.41, 4.47, 12
REASONING_TOKENS = {"minimal": 0, "none": 0, "low": 450, "medium": 900, "high": 1800}  # средние по замерам
STT_MODEL, TTS_MODEL, TTS_VOICES = "gpt-transcribe", "tts-1", {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}
VOICE_RPM, VOICE_MAX_FILE_BYTES, TTS_MAX_CHARS = 10, 5 * 1024 * 1024, 4096
# Цены выведены из документа организаторов: 3 доллара — это около 11 часов распознавания или 200 тыс. символов синтеза.
STT_DOLLARS_PER_MB, TTS_DOLLARS_PER_CHAR = 3 / (11 * 60 * 0.5), 3 / 200_000
FAKE_MP3 = b"ID3EMU-TTS:"  # заглушка синтеза: после метки идёт озвученный текст
SPEC = json.loads((Path(__file__).parent.parent / "docs/api/openapi.json").read_text(encoding="utf-8"))
ANSWERS_SCHEMA = SPEC["components"]["schemas"]["Case4Answers"]


def estimate_tokens(text: str) -> int:
    cyrillic = len(re.findall("[А-Яа-яЁё]", text))
    return round(cyrillic / CYRILLIC_CHARS_PER_TOKEN + (len(text) - cyrillic) / OTHER_CHARS_PER_TOKEN)


def error(status: int, code: str, message: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse({"code": code, "message": message, "request_id": "emu"}, status_code=status, headers=headers)


def create_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="Эмулятор стенда")
    datasets = {}
    for item in args.dataset:
        code, path = item.split("=", 1)
        datasets[code] = json.loads(Path(path).read_text(encoding="utf-8"))
    llm = CodexLLM(args.cache, parallel=args.parallel)
    starts: deque[float] = deque()
    rejected = set(filter(None, args.reject.split(",")))
    stats = {"llm_calls": 0, "tokens": args.spent, "http_429_minute": 0, "http_429_daily": 0, "faults": 0,
             "peak_requests_per_minute": 0, "getdata_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
             "stt_calls": 0, "tts_calls": 0, "stt_dollars": args.voice_spent, "tts_dollars": args.voice_spent,
             "http_402_voice": 0}
    voice_starts: dict[str, deque[float]] = {"stt": deque(), "tts": deque()}
    rng = random.Random(args.seed)

    def bearer(authorization: str) -> str:
        scheme, _, token = authorization.partition(" ")
        return token if scheme.lower() == "bearer" else ""

    # ---------- сервис хакатона ----------

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "datasets_loaded": len(datasets)}

    @app.get("/_emulator/stats")
    async def emulator_stats() -> dict:
        return stats

    @app.get("/case/4/getData")
    async def get_data(dataset_code: str | None = None, authorization: str = Header(default="")):
        token = bearer(authorization)
        if token not in (TEAM_TOKEN, RUN_TOKEN):
            return error(401, "UNAUTHORIZED", "Неверный или истёкший токен")
        if not dataset_code:
            if token == TEAM_TOKEN:
                return error(400, "BAD_REQUEST", "Нужен dataset_code")
            dataset_code = next(iter(datasets))  # токен прогона знает свой набор
        if dataset_code not in datasets:
            return error(403, "FORBIDDEN", "Нет доступа к набору")
        stats["getdata_calls"] += 1
        bundle = datasets[dataset_code]
        if token == RUN_TOKEN:  # закрытый набор: эталона нет, вид запросов не указан
            dataset = {**bundle["dataset"], "meta": {**bundle["dataset"]["meta"], "phase": "online"}}
            return {"dataset": dataset, "reference": None}
        return bundle

    @app.post("/case/4/validate")
    async def validate(request: Request, authorization: str = Header(default="")):
        token = bearer(authorization)
        if token == RUN_TOKEN:
            return error(403, "FORBIDDEN", "Метод доступен только для dev")
        if token != TEAM_TOKEN:
            return error(401, "UNAUTHORIZED", "Неверный или истёкший токен")
        try:
            body = await request.json()
        except ValueError:
            return error(400, "BAD_REQUEST", "Тело запроса не JSON")
        dataset = next(iter(datasets.values()))["dataset"]
        errors = [
            {"path": "/" + "/".join(map(str, e.absolute_path)), "record_id": None, "code": "SCHEMA", "message": e.message}
            for e in jsonschema.Draft202012Validator(ANSWERS_SCHEMA).iter_errors(body)
        ]
        if not errors:
            sections = {(a["article_id"], s["section_id"]) for a in dataset["articles"] for s in a["sections"]}
            answered = [a["record_id"] for a in body["requests"]]
            for request_item in dataset["requests"]:
                if answered.count(request_item["record_id"]) != 1:
                    errors.append({"path": "/requests", "record_id": request_item["record_id"], "code": "MISSING",
                                   "message": "Нужен ровно один ответ на запрос"})
            for n, answer in enumerate(body["requests"]):
                for citation in answer["citations"]:
                    if (citation["article_id"], citation["section_id"]) not in sections:
                        errors.append({"path": f"/requests/{n}/citations", "record_id": answer["record_id"],
                                       "code": "UNKNOWN_CITATION", "message": "Ссылка на несуществующий раздел"})
        return {"valid": not errors, "errors": errors}

    # ---------- прокси нейросети ----------

    def llm_error(status: int, message: str, headers: dict | None = None) -> JSONResponse:
        return JSONResponse({"error": {"message": message, "type": "emulator"}}, status_code=status, headers=headers)

    @app.get("/v1/models")
    async def models(authorization: str = Header(default="")):
        if bearer(authorization) != LLM_KEY:
            return llm_error(401, "invalid api key")
        return {"data": [{"id": f"openai/{args.model}", "object": "model"}]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request, authorization: str = Header(default="")):
        if bearer(authorization) != LLM_KEY:
            return llm_error(401, "invalid api key")
        body = await request.json()
        if body.get("model") != args.model:
            return llm_error(400, f"model {body.get('model')!r} is not available")
        unsupported = rejected & body.keys()
        if unsupported:
            return llm_error(400, f"Unsupported parameter: {sorted(unsupported)[0]}")
        now = time.monotonic()
        while starts and now - starts[0] >= 60:
            starts.popleft()
        if len(starts) >= args.rpm:
            stats["http_429_minute"] += 1
            return llm_error(429, "rate limit: requests per minute", {"Retry-After": str(int(60 - (now - starts[0])) + 1)})
        if stats["tokens"] >= args.budget:
            stats["http_429_daily"] += 1
            return llm_error(429, "rate limit: tokens per day", {"Retry-After": "3600"})
        starts.append(now)
        stats["peak_requests_per_minute"] = max(stats["peak_requests_per_minute"], len(starts))
        if rng.random() < args.fault_rate:
            stats["faults"] += 1
            if rng.random() < 0.5:
                return llm_error(503, "upstream unavailable")
            return JSONResponse({"choices": [], "usage": {"prompt_tokens": 0, "completion_tokens": 0}})

        messages = body.get("messages", [])
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        user = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
        effort = body.get("reasoning_effort", "low")
        limit = body.get("max_completion_tokens") or body.get("max_tokens") or 4096
        prompt_tokens = estimate_tokens(system) + estimate_tokens(user) + MESSAGE_OVERHEAD_TOKENS
        reasoning = REASONING_TOKENS.get(effort, 450)
        stats["llm_calls"] += 1
        if reasoning >= limit:  # лимит съеден рассуждением: настоящий прокси отвечает 200 без choices
            stats["tokens"] += prompt_tokens + limit
            return JSONResponse({"choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": limit}})
        try:
            content = json.dumps(await llm.chat_json(system, user, max_tokens=limit, effort=effort), ensure_ascii=False)
        except LLMError as failure:
            return llm_error(502, f"model call failed: {failure}")
        completion = estimate_tokens(content) + reasoning
        finish = "stop"
        if completion > limit:
            finish, completion = "length", limit
            content = content[: max(0, int((limit - reasoning) * CYRILLIC_CHARS_PER_TOKEN))]
        stats["tokens"] += prompt_tokens + completion
        stats["prompt_tokens"] += prompt_tokens
        stats["completion_tokens"] += completion
        return {
            "id": "emu", "object": "chat.completion", "model": args.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion,
                      "total_tokens": prompt_tokens + completion,
                      "prompt_tokens_details": {"cached_tokens": 0},
                      "completion_tokens_details": {"reasoning_tokens": reasoning}},
        }

    # ---------- голос: тот же ключ, общий лимит запросов в минуту, свой суточный бюджет ----------

    def voice_gate(kind: str, authorization: str) -> JSONResponse | None:
        """Проверки, общие для распознавания и синтеза. None — запрос принят и учтён."""
        if bearer(authorization) != LLM_KEY:
            return llm_error(401, "invalid api key")
        now = time.monotonic()
        for window in (starts, voice_starts[kind]):
            while window and now - window[0] >= 60:
                window.popleft()
        if len(starts) >= args.rpm or len(voice_starts[kind]) >= VOICE_RPM:
            stats["http_429_minute"] += 1
            return llm_error(429, "rate limit: requests per minute", {"Retry-After": "60"})
        if stats[f"{kind}_dollars"] >= args.voice_budget:
            stats["http_402_voice"] += 1
            return llm_error(402, f"daily {kind} budget exceeded")
        starts.append(now)
        voice_starts[kind].append(now)
        stats["peak_requests_per_minute"] = max(stats["peak_requests_per_minute"], len(starts))
        return None

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(request: Request, authorization: str = Header(default="")):
        body = await request.body()
        if len(body) > VOICE_MAX_FILE_BYTES + 4096:  # прокси не принимает файл больше 5 МБ
            return llm_error(413, "file is too large")
        refused = voice_gate("stt", authorization)
        if refused:
            return refused
        message = BytesParser(policy=HTTP).parsebytes(
            b"Content-Type: " + request.headers.get("content-type", "").encode() + b"\r\n\r\n" + body
        )
        parts = {part.get_param("name", header="content-disposition"): part for part in message.iter_parts()}
        field = lambda name: (parts[name].get_payload(decode=True) or b"").decode().strip() if name in parts else ""  # noqa: E731
        if field("model") != STT_MODEL:
            return llm_error(400, f"model {field('model')!r} is not available for transcription")
        if field("response_format") not in ("", "json"):
            return llm_error(400, "response_format must be json")
        filename = parts["file"].get_filename() if "file" in parts else None
        if not filename or filename.rsplit(".", 1)[-1] not in ("mp3", "wav", "webm", "m4a", "ogg", "flac"):
            return llm_error(400, "unsupported or missing file")
        audio = parts["file"].get_payload(decode=True) or b""
        stats["stt_calls"] += 1
        stats["stt_dollars"] += len(audio) / 1_000_000 * STT_DOLLARS_PER_MB
        text = audio[len(FAKE_MP3):].decode(errors="replace") if audio.startswith(FAKE_MP3) else args.voice_text
        return {"text": text}

    @app.post("/v1/audio/speech")
    async def speech(request: Request, authorization: str = Header(default="")):
        refused = voice_gate("tts", authorization)
        if refused:
            return refused
        body = await request.json()
        text = body.get("input")
        if body.get("model") != TTS_MODEL or body.get("voice") not in TTS_VOICES:
            return llm_error(400, "unknown model or voice")
        if not isinstance(text, str) or not text or len(text) > TTS_MAX_CHARS or body.get("response_format", "mp3") != "mp3":
            return llm_error(400, "input must be 1-4096 characters, response_format mp3")
        stats["tts_calls"] += 1
        stats["tts_dollars"] += len(text) * TTS_DOLLARS_PER_CHAR
        return Response(content=FAKE_MP3 + text.encode(), media_type="audio/mpeg")

    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", action="append", required=True,
                        help="код=путь к файлу ответа getData; первый — набор прогона")
    parser.add_argument("--cache", default="emulator", help="имя локального кеша ответов модели")
    parser.add_argument("--port", type=int, default=8900)
    parser.add_argument("--model", default=MODEL, help="имя модели, которое принимает прокси")
    parser.add_argument("--rpm", type=int, default=30)
    parser.add_argument("--budget", type=int, default=300_000, help="суточный лимит токенов")
    parser.add_argument("--spent", type=int, default=0, help="сколько токенов лимита уже потрачено до прогона")
    parser.add_argument("--reject", default="", help="параметры запроса, на которые прокси отвечает 400 (смена модели)")
    parser.add_argument("--fault-rate", type=float, default=0.0, help="доля вызовов, которые получают сбой")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--voice-budget", type=float, default=3.0,
                        help="суточный бюджет голоса в долларах, на распознавание и на синтез")
    parser.add_argument("--voice-spent", type=float, default=0.0, help="сколько бюджета голоса уже потрачено")
    parser.add_argument("--voice-text", default="Мешает ли чашка кофе после обеда уснуть вечером?",
                        help="что возвращает распознавание настоящей записи")
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args()
    uvicorn.run(create_app(arguments), host="0.0.0.0", port=arguments.port, log_level="warning")
