"""HTTP-вход серверной части: проверка живости и метод /run для стенда."""

import logging

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from app import config, hack_api, pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("app")

app = FastAPI(title="Mind-flowing · Кейс 4", docs_url=None, redoc_url=None)

# Последний полученный набор: после /run финального показа клиент работает с ним.
app.state.dataset = None


class RunRequest(BaseModel):
    case: int
    dataset_code: str | None = None


@app.get("/")
async def health() -> dict:
    return {"status": "ok", "case": config.CASE_ID}


@app.post("/run")
async def run(body: RunRequest, authorization: str = Header(default="")) -> dict:
    if body.case != config.CASE_ID:
        raise HTTPException(400, f"Решение рассчитано на кейс {config.CASE_ID}")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(401, "Нужен заголовок Authorization: Bearer <токен>")

    try:
        dataset = await hack_api.fetch_dataset(token, body.dataset_code)
    except hack_api.HackApiError as error:
        log.error("getData не удался: статус %s", error.status_code)
        raise HTTPException(502, "Не удалось получить набор у сервиса хакатона") from error

    app.state.dataset = dataset
    log.info("набор %s: запросов %d", dataset["meta"]["dataset_id"], len(dataset["requests"]))
    return await pipeline.answer_dataset(dataset)
