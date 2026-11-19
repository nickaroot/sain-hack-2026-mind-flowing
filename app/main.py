"""HTTP-вход серверной части: проверка живости, метод /run для стенда, методы /api и сам клиент по /app/."""

import asyncio
import hashlib
import logging
import time
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import JSONResponse
from starlette.staticfiles import StaticFiles

from app import chat, config, hack_api, logs, pipeline, state, voice
from app.contracts import RunResult

logs.setup()
log = logging.getLogger("app")

app = FastAPI(title="Mind-flowing · Кейс 4", docs_url=None, redoc_url=None)


class ApiBodyLimit:
    """Методы для клиента открыты всем, а памяти 512 МБ: тело запроса к /api не читается дальше предела."""

    def __init__(self, inner, max_bytes: int, larger: dict[str, int] | None = None):
        self.inner, self.default, self.larger = inner, max_bytes, larger or {}

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope["path"].startswith("/api"):
            return await self.inner(scope, receive, send)
        max_bytes = self.larger.get(scope["path"], self.default)  # запись голоса больше вопроса текстом
        declared = dict(scope["headers"]).get(b"content-length", b"")
        if declared.isdigit() and int(declared) > max_bytes:
            return await JSONResponse({"detail": "Слишком длинный запрос"}, status_code=413)(scope, receive, send)
        seen = 0

        async def limited():
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":  # длина не объявлена или объявлена неверно: считаем сами
                seen += len(message.get("body", b""))
                if seen > max_bytes:
                    raise HTTPException(413, "Слишком длинный запрос")
            return message

        await self.inner(scope, limited, send)


app.add_middleware(
    ApiBodyLimit, max_bytes=config.API_MAX_BODY_BYTES, larger={"/api/transcribe": config.VOICE_MAX_AUDIO_BYTES}
)
# Клиент на финальном показе открывают в браузере на компьютере организаторов, а серверная часть работает
# на стенде: это разные адреса, и без разрешения браузер не пустит запросы. Учётных данных в запросах нет.
# Добавлен последним, то есть стоит снаружи: заголовки получают и отказы предыдущего слоя.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
app.include_router(chat.router)
app.include_router(voice.router)
state.init(app)

# Готовая сборка клиента лежит в репозитории: на стенде нет Node, поэтому сервер отдаёт её как обычные файлы.
CLIENT_DIST = Path(__file__).resolve().parent.parent / "client" / "dist"


def mount_client(application: FastAPI, dist: Path = CLIENT_DIST) -> bool:
    """Отдаёт клиент по адресу /app/. Адреса / и /health, которыми стенд проверяет живость, не затронуты.

    Сборки нет — сервер работает как раньше: клиент для него необязателен, а ответы стенду важнее.
    """
    if not (dist / "index.html").is_file():
        return False
    application.mount("/app", StaticFiles(directory=dist, html=True), name="client")
    return True


mount_client(app)


MAX_KEPT_FALLBACK_SHARE = 0.1


class RunRequest(BaseModel):
    case: int
    dataset_code: str | None = None


@app.get("/")
@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "case": config.CASE_ID}


@app.post("/run")
async def run(body: RunRequest, authorization: str = Header(default="")) -> dict:
    # Тип ответа намеренно dict, а не RunResult: по точному типу FastAPI построил бы модель ответа и стал бы
    # пересобирать тело. Ответ стенду уходит ровно таким, каким его собрал конвейер.
    if body.case != config.CASE_ID:
        raise HTTPException(400, f"Решение рассчитано на кейс {config.CASE_ID}")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(401, "Нужен заголовок Authorization: Bearer <токен>")

    # Ключ — код набора вместе с отпечатком токена: другой токен или другой набор считаются заново.
    key = (body.dataset_code, hashlib.sha256(token.encode()).hexdigest())
    runs: dict[tuple, asyncio.Task] = app.state.runs
    task = runs.get(key)
    if task is None:
        chat.close_for_run(app)  # прогон важнее: разговоры, которые уже считаются или ждут очереди, прерываются
        task = runs[key] = asyncio.create_task(_answer(token, body.dataset_code))
    else:
        log.info("повторный /run: набор уже %s, второй раз не считается", "посчитан" if task.done() else "считается")

    def forget() -> None:
        if runs.get(key) is task:  # не трогаем прогон, который уже начал следующий запрос
            del runs[key]
    try:
        # shield: если вызывающая сторона оборвала соединение, прогон продолжается и достанется повторному запросу
        answers = await asyncio.shield(task)
    except hack_api.HackApiError as error:
        forget()
        log.error("getData не удался: статус %s", error.status_code)
        raise HTTPException(502, "Не удалось получить набор у сервиса хакатона") from error
    except asyncio.CancelledError:
        raise
    except Exception:
        forget()  # сбой не запоминаем: следующий запрос начнёт заново
        raise
    fallbacks = sum(pipeline.is_fallback(a) for a in answers["requests"])
    if fallbacks > MAX_KEPT_FALLBACK_SHARE * len(answers["requests"]):
        # Нейросеть была недоступна для заметной части набора: повторный /run стоит посчитать заново.
        forget()
    app.state.run_finished = True
    return answers


async def _answer(token: str, dataset_code: str | None) -> RunResult:
    started = time.monotonic()
    logs.start_run()  # метка прогона попадает во все строки логов этой задачи и её подзадач
    dataset = await hack_api.fetch_dataset(token, dataset_code)
    app.state.dataset = dataset
    log.info("набор получен: запросов %d, статей %d", len(dataset["requests"]), len(dataset["articles"]))
    llm = pipeline.LLM()
    remembered = app.state.responses.stats()["hits"]
    try:
        result = await pipeline.answer_dataset(
            dataset, llm, cache=app.state.responses if config.RUN_RESPONSE_CACHE else None
        )
    finally:
        chat.state_of(app).note_run(llm.usage.total_tokens)  # разговорам достаётся то, что осталось от прогона
    answers = result["requests"]
    log.info(
        "прогон завершён за %.0f с: ответов %d, из них отказов %d и запасных %d; токенов %d, "
        "вызовов нейросети %d, взято из памяти ответов %d",
        time.monotonic() - started, len(answers), sum(a["refused"] for a in answers),
        sum(pipeline.is_fallback(a) for a in answers), llm.usage.total_tokens,
        llm.usage.calls, app.state.responses.stats()["hits"] - remembered,
    )
    return result
