"""Голос для клиента: распознавание вопроса и озвучивание ответа эксперта.

Ключ прокси есть только у серверной части. Браузер записывает вопрос и присылает запись сюда, сервер
обращается к прокси и возвращает текст; так же в обратную сторону — текст ответа превращается в mp3.
У каждого эксперта свой голос: «эксперт как плейлист» слышно буквально.

Голос не тратит суточный лимит токенов нейросети, но его запросы входят в общие 30 запросов в минуту,
поэтому они идут через ту же очередь, что и вызовы нейросети, и уступают прогону набора.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx
from fastapi import APIRouter, FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from app import config, llm
from app.cache import BoundedCache, digest
from app.state import moscow_day, run_active

log = logging.getLogger("app.voice")
T = TypeVar("T")
router = APIRouter(prefix="/api")

# Голоса прокси: alloy, echo, fable, onyx, nova, shimmer. Эксперт без своего голоса говорит голосом по умолчанию.
VOICES = {"clinician": "onyx", "biohacker": "echo", "nutritionist": "nova"}
DEFAULT_VOICE = "alloy"
# Формат записи прокси определяет по расширению имени файла, поэтому оно выводится из типа содержимого.
AUDIO_EXTENSIONS = {
    "audio/webm": "webm", "audio/ogg": "ogg", "audio/mpeg": "mp3", "audio/mp3": "mp3", "audio/wav": "wav",
    "audio/x-wav": "wav", "audio/wave": "wav", "audio/mp4": "m4a", "audio/m4a": "m4a", "audio/x-m4a": "m4a",
    "audio/flac": "flac",
}
PARALLEL_REQUESTS = 2  # одновременно в памяти не больше двух записей по 5 МБ
MAX_WAITING = 6
SPEECH_CACHE_SIZE = 30  # mp3 ответа весит 100–200 КБ: повторное прослушивание не тратит бюджет
SPEECH_CACHE_BYTES = 16 * 1024 * 1024  # а ответ в 2000 символов — около 2 МБ: число записей память не ограничивает
HEARD_CACHE_SIZE = 50
RUN_IN_PROGRESS = "Идёт обработка набора, попробуйте через пару минут"


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1, max_length=config.VOICE_MAX_TEXT_CHARS)
    expert_id: str | None = Field(default=None, max_length=64)


class VoiceUnavailable(Exception):
    """Прокси не дал результата. Несёт то, что нужно сказать клиенту."""

    def __init__(self, status_code: int, detail: str, retry_after: int | None = None):
        super().__init__(detail)
        self.status_code, self.detail, self.retry_after = status_code, detail, retry_after

    def as_http(self) -> HTTPException:
        headers = {"Retry-After": str(self.retry_after)} if self.retry_after else None
        return HTTPException(self.status_code, self.detail, headers=headers)


class VoiceProxy:
    """Обращения к прокси за голосом. Протокол — как у OpenAI: /audio/transcriptions и /audio/speech."""

    async def transcribe(self, audio: bytes, extension: str, content_type: str) -> str:
        response = await self._post(
            "/audio/transcriptions",
            data={"model": config.STT_MODEL, "language": "ru", "response_format": "json"},
            files={"file": (f"question.{extension}", audio, content_type)},
        )
        try:
            return str(response.json()["text"]).strip()
        except (ValueError, KeyError, TypeError) as error:
            raise VoiceUnavailable(502, "Сервис распознавания ответил не по протоколу") from error

    async def speak(self, text: str, voice: str) -> bytes:
        response = await self._post(
            "/audio/speech",
            json={"model": config.TTS_MODEL, "voice": voice, "input": text, "response_format": "mp3"},
        )
        if not response.content:
            raise VoiceUnavailable(502, "Сервис синтеза вернул пустой ответ")
        return response.content

    async def _post(self, path: str, **body) -> httpx.Response:
        if not config.LLM_BASE_URL or not config.LLM_API_KEY:
            raise VoiceUnavailable(503, "Голос не настроен: нет адреса или ключа прокси")
        await llm.wait_turn()  # голос входит в общий лимит запросов в минуту
        try:
            async with httpx.AsyncClient(timeout=config.VOICE_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    f"{config.LLM_BASE_URL}{path}", headers={"Authorization": f"Bearer {config.LLM_API_KEY}"}, **body
                )
        except httpx.HTTPError as error:
            log.warning("голос: прокси недоступен (%s)", type(error).__name__)
            raise VoiceUnavailable(502, "Сервис голоса недоступен, повторите позже", retry_after=10) from error
        if response.status_code == 200:
            return response
        log.warning("голос: прокси ответил %s", response.status_code)
        if response.status_code == 402:  # суточный бюджет голоса у организаторов исчерпан до 00:00 по Москве
            raise VoiceUnavailable(429, "Голос на сегодня закончился", retry_after=3600)
        if response.status_code == 429:
            raise VoiceUnavailable(429, "Слишком много запросов, повторите через минуту", retry_after=60)
        if response.status_code == 413:
            raise VoiceUnavailable(413, "Запись слишком большая")
        raise VoiceUnavailable(502, "Сервис голоса ответил ошибкой, повторите позже", retry_after=10)


class VoiceState:
    """Очередь, суточные потолки и кеш озвученных ответов. Живёт в памяти процесса."""

    def __init__(self) -> None:
        self.gate = asyncio.Semaphore(PARALLEL_REQUESTS)
        self.waiting = 0
        self.speech: BoundedCache[bytes] = BoundedCache(
            "озвученные ответы", SPEECH_CACHE_SIZE, SPEECH_CACHE_BYTES, size_of=len
        )
        # Та же запись, присланная снова (повтор после сбоя сети у клиента), второй раз не распознаётся.
        self.heard: BoundedCache[str] = BoundedCache("распознанные записи", HEARD_CACHE_SIZE)
        self.in_work: dict[str, asyncio.Task] = {}  # ответы, которые озвучиваются сейчас или ждут очереди
        self.day = ""
        self.transcriptions = 0
        self.speech_chars = 0

    def _roll(self) -> None:
        today = moscow_day()
        if today != self.day:
            self.day, self.transcriptions, self.speech_chars = today, 0, 0

    def check_transcriptions(self) -> None:
        self._roll()
        if self.transcriptions >= config.VOICE_DAILY_TRANSCRIPTIONS:
            raise HTTPException(429, "Распознавание на сегодня закончилось", headers={"Retry-After": "3600"})

    def take_transcription(self) -> None:
        """Проверяет суточный потолок и сразу списывает: между проверкой и списанием нет ожиданий."""
        self.check_transcriptions()
        self.transcriptions += 1

    def take_speech(self, chars: int) -> None:
        self._roll()
        if self.speech_chars + chars > config.VOICE_DAILY_SPEECH_CHARS:
            raise HTTPException(429, "Озвучивание на сегодня закончилось", headers={"Retry-After": "3600"})
        self.speech_chars += chars


def state_of(app: FastAPI) -> VoiceState:
    if getattr(app.state, "voice", None) is None:
        app.state.voice = VoiceState()
    return app.state.voice


def _yield_to_run(app: FastAPI) -> None:
    if run_active(app):
        raise HTTPException(503, RUN_IN_PROGRESS, headers={"Retry-After": "60"})


async def _in_turn(app: FastAPI, state: VoiceState, work: Callable[[], Awaitable[T]]) -> T:
    """Выполняет обращение к прокси в очереди голоса; прогон набора и переполненная очередь — отказ.

    Суточный потолок work проверяет и списывает сам, уже дождавшись очереди: пока запрос ждал, потолок
    могли выбрать другие, а прогон набора — начаться. Запрос, уже ушедший к прокси, не отзывается:
    таких не больше PARALLEL_REQUESTS, и каждый занимает у прогона один старт в общей очереди.
    """
    _yield_to_run(app)
    if state.waiting >= MAX_WAITING:
        raise HTTPException(429, "Слишком много запросов голоса, повторите через несколько секунд", headers={"Retry-After": "10"})
    state.waiting += 1
    try:
        async with state.gate:
            _yield_to_run(app)
            return await work()
    except VoiceUnavailable as error:
        raise error.as_http() from error
    finally:
        state.waiting -= 1


@router.post("/transcribe")
async def transcribe(request: Request) -> dict:
    """Запись вопроса (тело запроса — сами байты, тип — в Content-Type) → текст."""
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    extension = AUDIO_EXTENSIONS.get(content_type)
    if extension is None:
        raise HTTPException(415, "Нужна аудиозапись: webm, ogg, mp3, wav, m4a или flac")
    state = state_of(request.app)
    state.check_transcriptions()  # ранний отказ: запись до 5 МБ незачем читать, если потолок уже выбран

    async def work() -> str:
        audio = await request.body()  # читается уже в очереди: в памяти не больше двух записей сразу
        if not audio:
            raise HTTPException(422, "Пустая запись")
        key = digest(content_type, audio)
        text = state.heard.get(key)
        if text is None:
            state.take_transcription()
            text = await VoiceProxy().transcribe(audio, extension, content_type)
            state.heard.put(key, text)
        return text

    return {"text": await _in_turn(request.app, state, work)}


async def _synthesize(app: FastAPI, state: VoiceState, key: str, text: str, voice: str) -> bytes:
    async def work() -> bytes:
        state.take_speech(len(text))
        return await VoiceProxy().speak(text, voice)

    audio = await _in_turn(app, state, work)
    state.speech.put(key, audio)
    return audio


def _settle(state: VoiceState, key: str, task: asyncio.Task) -> None:
    if state.in_work.get(key) is task:
        del state.in_work[key]
    if not task.cancelled():
        task.exception()  # ошибку получит обработчик запроса; здесь она только помечается прочитанной


@router.post("/speak")
async def speak(body: SpeakRequest, request: Request) -> Response:
    """Текст ответа → mp3 голосом выбранного эксперта."""
    state = state_of(request.app)
    voice = VOICES.get(body.expert_id or "", DEFAULT_VOICE)
    key = digest(voice, body.text)
    audio = state.speech.get(key)
    if audio is None:
        task = state.in_work.get(key)  # этот же ответ уже озвучивается: ждём его, а не платим второй раз
        if task is None:
            task = state.in_work[key] = asyncio.create_task(_synthesize(request.app, state, key, body.text, voice))
            task.add_done_callback(lambda done: _settle(state, key, done))
        # shield: если клиент не дождался, ответ доозвучится и достанется его повторному запросу
        audio = await asyncio.shield(task)
    return Response(content=audio, media_type="audio/mpeg", headers={"Cache-Control": "private, max-age=3600"})
