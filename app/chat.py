"""Методы для клиента: список экспертов, разговор с экспертом и текст источника.

Разговор строится тем же конвейером, что и ответы на /run: реплика превращается в набор из одной тройки.
Поэтому один вызов сразу даёт ответ всех трёх экспертов с общей ссылкой — клиент показывает выбранного
и может переключить эксперта без нового обращения к нейросети.

Методы открыты всем в интернете и делят с прогоном набора один суточный лимит нейросети, поэтому новый
вопрос считается, только если прошёл допуск: прогон не идёт, запас токенов есть, очередь не переполнена.
"""

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from app import config, hack_api, pipeline

log = logging.getLogger("app.chat")
router = APIRouter(prefix="/api")

MAX_TEXT_CHARS = 2000
MAX_HISTORY_TURNS = 20
MAX_ID_CHARS = 64
CACHE_SIZE = 200
PARALLEL_CHATS = 2
MAX_QUESTIONS_IN_WORK = 4  # считаются сейчас или ждут очереди; остальным — «попробуйте чуть позже»
AGENT_ROLE = "agent"  # так реплики приложения названы в переписке набора
MOSCOW = timezone(timedelta(hours=3))  # суточный лимит прокси сбрасывается в 00:00 по Москве
RUN_IN_PROGRESS = "Идёт обработка набора, попробуйте через пару минут"


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)


class ChatRequest(BaseModel):
    expert_id: str = Field(min_length=1, max_length=MAX_ID_CHARS)
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)
    history: list[Turn] = Field(default_factory=list, max_length=MAX_HISTORY_TURNS)
    user_id: str | None = Field(default=None, max_length=MAX_ID_CHARS)


class ChatState:
    """Всё, что разговоры хранят между запросами. Живёт в памяти процесса: перезапуск обнуляет учёт."""

    def __init__(self) -> None:
        self.loading = asyncio.Lock()
        self.gate = asyncio.Semaphore(PARALLEL_CHATS)
        self.cache: OrderedDict[str, dict] = OrderedDict()
        self.in_work: dict[str, asyncio.Task] = {}  # вопросы, которые считаются или ждут очереди
        self.dev_dataset: dict | None = None  # набор разработки для клиента, пока стенд не вызвал /run
        self.day = ""
        self.chat_tokens = 0  # потрачено разговорами за сутки
        self.run_tokens = 0  # потрачено прогонами набора за сутки
        self.reserved = 0  # отложено под вопросы, которые ещё считаются

    def _roll(self) -> None:
        today = datetime.now(MOSCOW).date().isoformat()
        if today != self.day:
            self.day, self.chat_tokens, self.run_tokens = today, 0, 0

    def note_run(self, tokens: int) -> None:
        self._roll()
        self.run_tokens += tokens

    def note_chat(self, tokens: int) -> None:
        self._roll()
        self.chat_tokens += tokens

    def has_tokens_for_question(self, run_finished: bool) -> bool:
        self._roll()
        planned = self.chat_tokens + self.reserved + config.CHAT_TOKEN_ESTIMATE
        if planned > (config.CHAT_TOKEN_BUDGET_AFTER_RUN if run_finished else config.CHAT_TOKEN_BUDGET):
            return False
        return self.run_tokens + planned <= config.DAILY_TOKEN_LIMIT - config.DAILY_TOKEN_MARGIN


def state_of(app: FastAPI) -> ChatState:
    if getattr(app.state, "chat", None) is None:
        app.state.chat = ChatState()
    return app.state.chat


def _run_active(app: FastAPI) -> bool:
    return any(not task.done() for task in app.state.runs.values())


def close_for_run(app: FastAPI) -> None:
    """Вызывается перед прогоном набора: вопросы, которые считаются или ждут очереди, прерываются."""
    for task in list(state_of(app).in_work.values()):
        task.cancel()


def _frozen(app: FastAPI) -> bool:
    """Окно перед зачётным прогоном, в которое новые вопросы не считаются."""
    if app.state.run_finished or not config.CHAT_FREEZE_FROM or not config.CHAT_FREEZE_UNTIL:
        return False
    now = datetime.now(timezone.utc)
    return datetime.fromisoformat(config.CHAT_FREEZE_FROM) <= now < datetime.fromisoformat(config.CHAT_FREEZE_UNTIL)


async def _dataset(request: Request) -> dict:
    """Набор последнего /run, а пока прогона не было — набор разработки по токену команды."""
    app = request.app
    if app.state.dataset is not None:
        return app.state.dataset
    state = state_of(app)
    if state.dev_dataset is None:
        async with state.loading:
            if state.dev_dataset is None:
                try:
                    state.dev_dataset = await hack_api.fetch_dataset(config.HACK_TEAM_TOKEN, config.DEV_DATASET_CODE)
                except Exception as error:
                    log.error("набор для клиента не получен: %s", type(error).__name__)
                    raise HTTPException(503, "База статей ещё не загружена, попробуйте позже") from error
    # Пока шла загрузка, стенд мог вызвать /run: его набор главнее и набором разработки не затирается.
    return app.state.dataset if app.state.dataset is not None else state.dev_dataset


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


def _admit(app: FastAPI, state: ChatState) -> None:
    """Можно ли считать новый вопрос. Пропускает — и сразу резервирует под него токены."""
    if _run_active(app):
        # Прогон набора и разговоры делят лимиты прокси: прогон важнее.
        raise HTTPException(503, RUN_IN_PROGRESS, headers={"Retry-After": "60"})
    if _frozen(app):
        raise HTTPException(503, "Перед зачётным прогоном новые вопросы не принимаются", headers={"Retry-After": "3600"})
    if len(state.in_work) >= MAX_QUESTIONS_IN_WORK:
        raise HTTPException(429, "Эксперты заняты, повторите вопрос через несколько секунд", headers={"Retry-After": "15"})
    if not state.has_tokens_for_question(app.state.run_finished):
        raise HTTPException(429, "На сегодня запас разговоров исчерпан", headers={"Retry-After": "3600"})
    state.reserved += config.CHAT_TOKEN_ESTIMATE


async def _compute(app: FastAPI, state: ChatState, key: str, dataset: dict, body: ChatRequest, user_id: str) -> dict:
    """Считает ответ тройки на один вопрос. Токены под него уже зарезервированы в _admit."""
    llm = None
    try:
        async with state.gate:
            if _run_active(app):  # прогон начался, пока вопрос ждал очереди
                raise HTTPException(503, RUN_IN_PROGRESS, headers={"Retry-After": "60"})
            llm = pipeline.LLM()
            answers = await pipeline.answer_dataset(
                _live_dataset(dataset, body, user_id), llm, deadline_seconds=config.CHAT_DEADLINE_SECONDS,
                quick=True, history_messages=config.CHAT_HISTORY_MESSAGES,
            )
    finally:
        state.reserved -= config.CHAT_TOKEN_ESTIMATE
        if llm is not None:
            state.note_chat(llm.usage.total_tokens)
    voices = {
        a["record_id"]: {"text": a["text"], "refused": a["refused"], "fallback": pipeline.is_fallback(a)}
        for a in answers["requests"]
    }
    first = answers["requests"][0]
    triple = {
        "answers": voices,
        "citations": [
            {k: v for k, v in (_source(dataset, c["article_id"], c["section_id"]) or c).items() if k != "text"}
            for c in first["citations"]
        ],
    }
    if not any(voice["fallback"] for voice in voices.values()):  # запасной текст не запоминаем: повтор может удаться
        state.cache[key] = triple
        while len(state.cache) > CACHE_SIZE:
            state.cache.popitem(last=False)
    return triple


def _settle(state: ChatState, key: str, task: asyncio.Task) -> None:
    if state.in_work.get(key) is task:
        del state.in_work[key]
    if not task.cancelled():
        task.exception()  # ошибку уже получил или получит обработчик запроса; здесь она только помечается прочитанной


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
    started = time.monotonic()
    app, state = request.app, state_of(request.app)
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
        task = state.in_work.get(key)  # тот же вопрос уже считается: ждём его, а не считаем второй раз
        if task is None:
            _admit(app, state)
            task = state.in_work[key] = asyncio.create_task(_compute(app, state, key, dataset, body, user_id))
            task.add_done_callback(lambda done: _settle(state, key, done))
        try:
            # shield: если клиент не дождался, вопрос досчитается и достанется его повторному запросу
            triple = await asyncio.wait_for(
                asyncio.shield(task), timeout=max(1.0, config.CHAT_TIMEOUT_SECONDS - (time.monotonic() - started))
            )
        except asyncio.TimeoutError:
            raise HTTPException(504, "Эксперт не успел ответить, повторите вопрос", headers={"Retry-After": "5"}) from None
        except asyncio.CancelledError:
            if task.cancelled():  # вопрос прерван начавшимся прогоном набора
                raise HTTPException(503, RUN_IN_PROGRESS, headers={"Retry-After": "60"}) from None
            raise
    chosen = triple["answers"][body.expert_id]
    return {
        "expert_id": body.expert_id, "text": chosen["text"], "refused": chosen["refused"],
        # Нейросеть не ответила: человеку ушёл запасной текст. Клиент может предложить повторить вопрос.
        "degraded": chosen["fallback"],
        "answers": {expert_id: {"text": v["text"], "refused": v["refused"]} for expert_id, v in triple["answers"].items()},
        "citations": triple["citations"],
    }
