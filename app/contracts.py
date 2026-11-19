"""Формы данных на границах решения: набор организаторов на входе и ответ /run на выходе.

Это описания для чтения и для проверки типов (mypy). Во время работы данные остаются обычными словарями
в том виде, в каком пришли: ничего не копируется и не преобразуется, порядок полей ответа задаёт код.
Перечислены только поля, которые решение читает; остальные поля набора проходят мимо нетронутыми.
"""

from typing import NotRequired, TypedDict


class SectionData(TypedDict):
    section_id: str
    title: str
    text: str


class ArticleData(TypedDict):
    article_id: str
    title: str
    sections: list[SectionData]
    year: NotRequired[int | None]
    doi: NotRequired[str | None]


class ExpertData(TypedDict):
    expert_id: str
    role: str  # «Сомнолог-клиницист. Думает как врач…»: название и описание роли одним текстом
    speech_style: str


class Utterance(TypedDict):
    role: str
    at: str
    text: str


class RequestData(TypedDict):
    """Один запрос набора: реплика пользователя, на которую отвечает один эксперт."""

    record_id: str
    user_id: str
    expert_id: str
    group_id: NotRequired[str | None]  # у трёх экспертов, отвечающих на одну реплику, группа общая
    utterance: Utterance


class Message(TypedDict):
    """Реплика прежней переписки пользователя с приложением."""

    user_id: str
    role: str
    at: str
    text: str


class UserData(TypedDict):
    user_id: str


class Meta(TypedDict, total=False):
    dataset_id: str


class Dataset(TypedDict):
    articles: list[ArticleData]
    experts: list[ExpertData]
    requests: list[RequestData]
    chat_history: NotRequired[list[Message]]
    users: NotRequired[list[UserData]]
    meta: NotRequired[Meta]


class Citation(TypedDict):
    article_id: str
    section_id: str


class Answer(TypedDict):
    record_id: str
    text: str
    refused: bool
    citations: list[Citation]


class RunResult(TypedDict):
    """Тело ответа /run: ровно один ответ на каждый запрос набора, в порядке запросов."""

    requests: list[Answer]
