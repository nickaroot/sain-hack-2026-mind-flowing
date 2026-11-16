"""HTTP-вход серверной части: проверка живости и метод /run для стенда."""

import asyncio
import logging

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from app import config, hack_api, pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("app")
# httpx на уровне INFO пишет URL запроса, а в нём код набора: в логи он попадать не должен.
logging.getLogger("httpx").setLevel(logging.WARNING)

app = FastAPI(title="Mind-flowing · Кейс 4", docs_url=None, redoc_url=None)

# Последний полученный набор: после /run финального показа клиент работает с ним.
app.state.dataset = None
# Прогоны по коду набора. Повторный /run с тем же кодом (например, после обрыва связи у вызывающей стороны)
# получает тот же результат и не тратит суточный лимит токенов второй раз.
app.state.runs = {}


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

    runs: dict[str | None, asyncio.Task] = app.state.runs
    task = runs.get(body.dataset_code)
    if task is None:
        task = runs[body.dataset_code] = asyncio.create_task(_answer(token, body.dataset_code))
    try:
        # shield: если вызывающая сторона оборвала соединение, прогон продолжается и достанется повторному запросу
        return await asyncio.shield(task)
    except hack_api.HackApiError as error:
        runs.pop(body.dataset_code, None)
        log.error("getData не удался: статус %s", error.status_code)
        raise HTTPException(502, "Не удалось получить набор у сервиса хакатона") from error
    except asyncio.CancelledError:
        raise
    except Exception:
        runs.pop(body.dataset_code, None)  # сбой не запоминаем: следующий запрос начнёт заново
        raise


async def _answer(token: str, dataset_code: str | None) -> dict:
    dataset = await hack_api.fetch_dataset(token, dataset_code)
    app.state.dataset = dataset
    log.info("набор получен: запросов %d", len(dataset["requests"]))
    return await pipeline.answer_dataset(dataset)
