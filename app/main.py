"""HTTP-вход серверной части: проверка живости, метод /run для стенда и методы /api для клиента."""

import asyncio
import hashlib
import logging

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app import chat, config, hack_api, pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("app")
# httpx на уровне INFO пишет URL запроса, а в нём код набора: в логи он попадать не должен.
logging.getLogger("httpx").setLevel(logging.WARNING)

app = FastAPI(title="Mind-flowing · Кейс 4", docs_url=None, redoc_url=None)
# Клиент на финальном показе открывают в браузере на компьютере организаторов, а серверная часть работает
# на стенде: это разные адреса, и без разрешения браузер не пустит запросы. Учётных данных в запросах нет.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
app.include_router(chat.router)

# Последний полученный набор: после /run финального показа клиент работает с ним.
app.state.dataset = None
# Прогоны по коду набора. Повторный /run с тем же кодом (например, после обрыва связи у вызывающей стороны)
# получает тот же результат и не тратит суточный лимит токенов второй раз.
app.state.runs = {}
# Прошёл ли в этом процессе прогон набора: после него разговорам в клиенте отдаётся больший запас токенов.
app.state.run_finished = False
app.state.chat = None


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
        task = runs[key] = asyncio.create_task(_answer(token, body.dataset_code))

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


async def _answer(token: str, dataset_code: str | None) -> dict:
    dataset = await hack_api.fetch_dataset(token, dataset_code)
    app.state.dataset = dataset
    log.info("набор получен: запросов %d", len(dataset["requests"]))
    return await pipeline.answer_dataset(dataset)
