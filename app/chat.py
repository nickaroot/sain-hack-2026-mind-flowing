"""Методы для клиента: список экспертов, разговор с экспертом и текст источника.

Разговор строится тем же конвейером, что и ответы на /run: реплика превращается в набор из одной тройки.
Поэтому один вызов сразу даёт ответ всех трёх экспертов с общей ссылкой — клиент показывает выбранного
и может переключить эксперта без нового обращения к нейросети.
"""

import asyncio
import hashlib
import json
import logging
from collections import OrderedDict
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app import config, hack_api, pipeline

log = logging.getLogger("app.chat")
router = APIRouter(prefix="/api")

MAX_TEXT_CHARS = 2000
MAX_HISTORY_TURNS = 20
CACHE_SIZE = 200
PARALLEL_CHATS = 2  # вместе с паузой между вызовами держит разговоры далеко от 30 запросов в минуту
AGENT_ROLE = "agent"  # так реплики приложения названы в переписке набора


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)


class ChatRequest(BaseModel):
    expert_id: str
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    history: list[Turn] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)
    user_id: str | None = None


class ChatState:
    """Всё, что разговоры хранят между запросами. Живёт в памяти процесса."""

    def __init__(self) -> None:
        self.loading = asyncio.Lock()
        self.gate = asyncio.Semaphore(PARALLEL_CHATS)
        self.cache: OrderedDict[str, dict] = OrderedDict()
        self.spent_day = ""
        self.spent_tokens = 0

    def spent_today(self) -> int:
        today = datetime.now(timezone.utc).date().isoformat()
        if today != self.spent_day:
            self.spent_day, self.spent_tokens = today, 0
        return self.spent_tokens


def _state(request: Request) -> ChatState:
    state = request.app.state
    if getattr(state, "chat", None) is None:
        state.chat = ChatState()
    return state.chat


async def _dataset(request: Request) -> dict:
    """Набор последнего /run, а пока прогона не было — набор разработки по токену команды."""
    state = request.app.state
    if state.dataset is not None:
        return state.dataset
    async with _state(request).loading:
        if state.dataset is None:
            try:
                state.dataset = await hack_api.fetch_dataset(config.HACK_TEAM_TOKEN, config.DEV_DATASET_CODE)
            except Exception as error:
                log.error("набор для клиента не получен: %s", type(error).__name__)
                raise HTTPException(503, "База статей ещё не загружена, попробуйте позже") from error
    return state.dataset


def _default_user(dataset: dict) -> str | None:
    users = dataset.get("users") or []
    return users[0]["user_id"] if len(users) == 1 else None


def _expert_card(expert: dict) -> dict:
    # Описание роли в наборе начинается с названия: «Сомнолог-клиницист. Думает как врач…»
    name, _, description = expert.get("role", "").partition(". ")
    return {
        "expert_id": expert["expert_id"], "name": name.rstrip(".") or expert["expert_id"],
        "description": description, "speech_style": expert.get("speech_style", ""),
    }


def _source(dataset: dict, article_id: str, section_id: str) -> dict | None:
    for article in dataset["articles"]:
        if article["article_id"] != article_id:
            continue
        for section in article.get("sections", []):
            if section["section_id"] == section_id:
                return {
                    "article_id": article_id, "section_id": section_id,
                    "article_title": article.get("title", ""), "section_title": section.get("title", ""),
                    "year": article.get("year"), "doi": article.get("doi"), "text": section.get("text", ""),
                }
    return None


def _live_dataset(dataset: dict, body: ChatRequest, user_id: str) -> dict:
    """Набор из одной тройки: реплика человека для каждого эксперта и переписка, включая текущий разговор."""
    now = datetime.now(timezone.utc)
    turns = [
        {"user_id": user_id, "role": "user" if turn.role == "user" else AGENT_ROLE, "text": turn.text,
         "at": (now - timedelta(seconds=len(body.history) - i)).isoformat()}
        for i, turn in enumerate(body.history)
    ]
    utterance = {"role": "user", "at": now.isoformat(), "text": body.text}
    return {
        **dataset,
        "chat_history": [*dataset.get("chat_history", []), *turns],
        "requests": [
            {"record_id": expert["expert_id"], "user_id": user_id, "expert_id": expert["expert_id"],
             "group_id": "live", "utterance": utterance}
            for expert in dataset["experts"]
        ],
    }


@router.get("/experts")
async def experts(request: Request) -> dict:
    dataset = await _dataset(request)
    return {"experts": [_expert_card(e) for e in dataset["experts"]], "user_id": _default_user(dataset)}


@router.get("/sources/{article_id}/{section_id}")
async def source(article_id: str, section_id: str, request: Request) -> dict:
    found = _source(await _dataset(request), article_id, section_id)
    if found is None:
        raise HTTPException(404, "Такого раздела в базе нет")
    return found


@router.post("/chat")
async def chat(body: ChatRequest, request: Request) -> dict:
    app_state, state = request.app.state, _state(request)
    if any(not task.done() for task in app_state.runs.values()):
        # Прогон набора и разговоры делят один лимит запросов в минуту: прогон важнее.
        raise HTTPException(503, "Идёт обработка набора, попробуйте через пару минут", headers={"Retry-After": "60"})
    dataset = await _dataset(request)
    if body.expert_id not in {e["expert_id"] for e in dataset["experts"]}:
        raise HTTPException(404, "Такого эксперта нет")
    user_id = body.user_id or _default_user(dataset) or "live-user"

    key = hashlib.sha256(json.dumps(
        [dataset.get("meta", {}).get("dataset_id"), user_id, body.text, [t.model_dump() for t in body.history]],
        ensure_ascii=False,
    ).encode()).hexdigest()
    triple = state.cache.get(key)
    if triple is None:
        budget = config.CHAT_TOKEN_BUDGET_AFTER_RUN if getattr(app_state, "run_finished", False) else config.CHAT_TOKEN_BUDGET
        if state.spent_today() >= budget:
            raise HTTPException(429, "На сегодня запас разговоров исчерпан", headers={"Retry-After": "3600"})
        async with state.gate:
            llm = pipeline.LLM()
            try:
                answers = await pipeline.answer_dataset(
                    _live_dataset(dataset, body, user_id), llm, deadline_seconds=config.CHAT_DEADLINE_SECONDS, quick=True
                )
            finally:
                state.spent_today()
                state.spent_tokens += llm.usage.total_tokens
        by_expert = {a["record_id"]: a for a in answers["requests"]}
        first = next(iter(by_expert.values()))
        triple = {
            "answers": {expert_id: a["text"] for expert_id, a in by_expert.items()},
            "refused": first["refused"],
            "citations": [
                {k: v for k, v in (_source(dataset, c["article_id"], c["section_id"]) or c).items() if k != "text"}
                for c in first["citations"]
            ],
            # Нейросеть не ответила: человеку ушёл запасной текст. Клиент может предложить повторить вопрос.
            "degraded": any(pipeline.is_fallback(a) for a in by_expert.values()),
        }
        if not triple["degraded"]:
            state.cache[key] = triple
            while len(state.cache) > CACHE_SIZE:
                state.cache.popitem(last=False)
    return {"expert_id": body.expert_id, "text": triple["answers"][body.expert_id], **triple}
