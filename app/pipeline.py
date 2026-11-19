"""Сборка ответов на набор кейса 4.

Порядок работы:
1. Один вызов нейросети на пачку до 20 реплик: английские поисковые слова и статьи, выбранные по списку названий.
2. BM25 находит разделы статей для каждой тройки запросов; порядок статей объединяется с выбором нейросети.
3. Один вызов на тройку: проверка на недопустимую просьбу, выбор источника и три текста экспертов.
   Общая ссылка у тройки получается сама собой, а запросы разных пользователей не смешиваются.
4. Тексты проверяет код (app/text_checks.py). С замечаниями делается одна повторная попытка.
   Если нейросеть недоступна или текст небезопасен, запись получает запасной отказ:
   без ответа не остаётся ни одна запись.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime

from app import config, prompts
from app.cache import ResponseCache
from app.contracts import Answer, ArticleData, Citation, Dataset, Message, RequestData, RunResult
from app.llm import LLM, JsonModel, LLMError
from app.retrieval import Hit, Index
from app.text_checks import is_safe_text, text_problems, tidy

log = logging.getLogger("app.pipeline")

CANDIDATES = 6
SNIPPET_CHARS = 620
ABOUT_CHARS = 260
HISTORY_MESSAGES = 4
HISTORY_CHARS = 500
QUERY_BATCH = 20

# Запасной отказ уходит, когда нейросеть не ответила или её текст небезопасен. В нём нет ни указаний, ни
# планов проверки: настоящая проверка медицинских формулировок помечала их рядом с отказом как лечение.
FALLBACK_TEXTS = {
    "clinician": "На этот вопрос я не отвечу: он требует очного разговора со специалистом. "
    "Из исследований о сне можно взять один безопасный вывод — он в разделе статьи по ссылке.",
    "biohacker": "Это не дам: тут нужен врач, а не эксперимент. Что исследования говорят о сне "
    "и привычках — в разделе статьи по ссылке.",
    "nutritionist": "Здесь я не подскажу, с этим лучше к врачу. А о том, как привычки связаны со сном, "
    "написано в разделе статьи по ссылке.",
}
REWRITE_NOTE = (
    "Прошлая попытка возвращена с замечаниями. Перепиши все три текста, исправь это и сохрани манеру "
    "каждого эксперта. "
)
REWRITE_NOTE_PROBLEMS_CHARS = 900


@dataclass
class SearchPlan:
    """Что нейросеть вернула для поиска по одной реплике."""

    terms: str = ""
    articles: list[str] = field(default_factory=list)


@dataclass
class Draft:
    """Одна попытка ответа на тройку и то, что о ней сказала проверка кодом."""

    texts: dict
    source: int  # номер источника в списке, который видела нейросеть, с единицы
    refused: bool
    unsafe: int  # сколько текстов придётся заменить запасным отказом
    problems: list[str]  # все замечания с именем эксперта: они уходят нейросети при повторной попытке
    serious: int = 0  # сколько текстов с замечаниями по существу, а не к длине и подаче

    @property
    def rank(self) -> tuple[int, int]:
        # Небезопасный текст заменяется запасным отказом и очков не приносит, поэтому безопасность важнее.
        return self.unsafe, len(self.problems)


@dataclass
class Triple:
    """Одна реплика и всё, что по ней собрано: запросы экспертов, найденные разделы, вход нейросети, попытка."""

    requests: list[RequestData]
    hits: list[Hit]
    payload: dict
    draft: Draft | None = None


def is_fallback(answer: Answer) -> bool:
    return answer["text"] in FALLBACK_TEXTS.values()


def _citation(hit: Hit) -> Citation:
    return {"article_id": hit.section.article_id, "section_id": hit.section.section_id}


def _moment(value: object) -> datetime:
    """Отметка времени как момент. Без часового пояса считаем её UTC; нечитаемую ставим в самое начало."""
    try:
        moment = datetime.fromisoformat(str(value))
    except ValueError:
        return datetime.min.replace(tzinfo=UTC)
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def _history_for(
    request: RequestData, by_user: dict[str, list[Message]], messages: int = HISTORY_MESSAGES
) -> list[str]:
    # Время сравниваем как моменты, а не как строки: у сообщений бывают разные часовые пояса.
    asked_at = _moment(request["utterance"].get("at"))
    earlier = [m for m in by_user.get(request["user_id"], []) if _moment(m.get("at")) < asked_at]
    return [f"{m['role']}: {m['text'][:HISTORY_CHARS]}" for m in earlier[-messages:]]


def _histories(chat_history: list[Message], groups: list[list[RequestData]], messages: int) -> list[list[str]]:
    """Для каждой тройки — последние реплики переписки того же пользователя до её вопроса."""
    by_user: dict[str, list[Message]] = defaultdict(list)
    for message in sorted(chat_history, key=lambda m: _moment(m.get("at"))):
        by_user[message["user_id"]].append(message)
    return [_history_for(group[0], by_user, messages) for group in groups]


def _group_requests(requests: list[RequestData]) -> list[list[RequestData]]:
    """Тройки по group_id; запрос без группы обрабатывается отдельно."""
    groups: dict[str, list[RequestData]] = defaultdict(list)
    for request in requests:
        groups[request.get("group_id") or f"single:{request['record_id']}"].append(request)
    return list(groups.values())


async def _search_plans(
    llm: JsonModel, groups: list[list[RequestData]], histories: list[list[str]], articles: list[ArticleData],
    cache: ResponseCache | None = None,
) -> list[SearchPlan]:
    """Для каждой группы — английские поисковые слова и статьи, выбранные по списку названий.

    Без плана тройка получает запасной ответ, поэтому сбой вызова не должен стоить целой пачки реплик:
    пачка делится пополам и запрашивается заново, пока не останется одна реплика.
    """
    plans = [SearchPlan() for _ in groups]
    known = {a["article_id"] for a in articles}
    catalog = (
        [{"id": a["article_id"], "title": a["title"]} for a in articles]
        if len(articles) <= config.CATALOG_MAX_ARTICLES
        else []
    )

    async def one_batch(batch: list[int]) -> None:
        if not batch:  # при делении пачки с одной недостающей репликой вторая половина пуста: вызов не нужен
            return
        items = []
        for i in batch:
            context = [h[len("user: "):] for h in histories[i] if h.startswith("user:")][-2:]
            items.append({"id": str(i), "context": context, "text": groups[i][0]["utterance"]["text"]})
        user = json.dumps({"catalog": catalog, "items": items}, ensure_ascii=False)
        cache_key = ResponseCache.key(config.LLM_MODEL, prompts.QUERY_SYSTEM, user, 4000, "minimal") if cache is not None else ""
        result = cache.get(cache_key) if cache is not None else None
        remembered = result is not None
        if result is None:
            try:
                result = await llm.chat_json(prompts.QUERY_SYSTEM, user, max_tokens=4000, effort="minimal")
            except LLMError:
                return  # вызов уже повторялся внутри LLM; дробить пачку при недоступной нейросети — только тратить время
        for key, value in result.items():
            if not (key.isdigit() and int(key) < len(plans)):
                continue
            if isinstance(value, str):  # нейросеть вернула только слова — поиск всё равно сработает
                value = {"terms": value}
            if not isinstance(value, dict) or not isinstance(value.get("terms"), str):
                continue
            named = value.get("articles")
            chosen = named if isinstance(named, list) else []
            plans[int(key)] = SearchPlan(
                value["terms"][:400], [a for a in chosen if isinstance(a, str) and a in known][:4]
            )
        missing = [i for i in batch if not plans[i].terms]
        if cache is not None and not remembered and not missing and set(result) == {str(i) for i in batch}:
            # Запоминается только полный план ровно на эту пачку: неполный при повторе стоит спросить заново.
            cache.put(cache_key, result)
        if missing and len(batch) > 1 and llm.usage.total_tokens < config.RETRY_TOKEN_BUDGET:
            half = (len(missing) + 1) // 2
            await asyncio.gather(one_batch(missing[:half]), one_batch(missing[half:]))

    order = list(range(len(groups)))
    await asyncio.gather(*(one_batch(order[start : start + QUERY_BATCH]) for start in range(0, len(order), QUERY_BATCH)))
    return plans


def _sources(index: Index, hits: list[Hit], query: str) -> list[dict]:
    """Кандидаты для нейросети. У первого раздела каждой статьи есть вырезка из её аннотации:
    по одному разделу из середины статьи не всегда видно, что статья именно о том, о чём спросили."""
    sources, described = [], set()
    for n, hit in enumerate(hits, start=1):
        section = hit.section
        source = {"n": n, "article": section.article_title, "section": section.title,
                  "text": index.focus(section.text, query, SNIPPET_CHARS)}
        abstract = index.abstract_of(section.article_id)
        if section.article_id not in described and abstract is not None and abstract.section_id != section.section_id:
            source["about"] = index.focus(abstract.text, query, ABOUT_CHARS)
        described.add(section.article_id)
        sources.append(source)
    return sources


async def _attempt(llm: JsonModel, system: str, triple: Triple, cache: ResponseCache | None = None) -> Draft | None:
    """Один вызов на тройку. None — нейросеть не ответила или вернула не то.

    Ответ запоминается в cache, только если к нему нет ни одного замечания, то есть он и есть итог тройки.
    Ответ не по форме, небезопасный или с замечаниями при повторе спрашивается заново.
    """
    user = json.dumps(triple.payload, ensure_ascii=False)
    cache_key = ResponseCache.key(config.LLM_MODEL, system, user, 3500, "low") if cache is not None else ""
    result = cache.get(cache_key) if cache is not None else None
    remembered = result is not None
    if result is None:
        try:
            result = await llm.chat_json(system, user, max_tokens=3500, effort="low")
        except LLMError:
            return None
    texts, source = result.get("texts"), result.get("source")
    if not isinstance(texts, dict) or not isinstance(source, int) or not 1 <= source <= len(triple.hits):
        return None
    refused = result.get("kind") == "refusal" or result.get("forbidden") not in (None, "none")
    unsupported = result.get("forbidden") == "unsupported"
    problems, unsafe, serious = [], 0, 0
    for request in triple.requests:
        text = texts.get(request["expert_id"])
        if not isinstance(text, str):
            problems.append(f"{request['expert_id']}: текста нет")
            unsafe += 1
            serious += 1
            continue
        found = text_problems(text, refused, unsupported)
        problems += [f"{request['expert_id']}: {problem.message}" for problem in found]
        unsafe += not is_safe_text(text)
        serious += any(problem.substantive for problem in found)
    if cache is not None and not remembered and not problems and not unsafe and type(source) is int:
        cache.put(cache_key, result)
    return Draft(texts, source, refused, unsafe, problems, serious)


def _better(first: Draft | None, second: Draft | None) -> Draft | None:
    """Какую из двух попыток оставить."""
    if first is None or second is None:
        return first or second
    if second.refused != first.refused:
        return first  # замечания касались текста, а не решения об отказе: повтор, который его поменял, не берём
    return second if second.rank <= first.rank else first  # при равенстве — уже исправленная попытка


def _needs_rewrite(draft: Draft | None, quick: bool) -> bool:
    if draft is None:
        return True
    return bool(draft.unsafe or draft.serious) if quick else bool(draft.problems)


def _rewrite_priority(draft: Draft | None) -> tuple[int, int, int]:
    """Кого переписывать первым, если лимита не хватит на всех: тройки без ответа, затем с небезопасным
    текстом, затем с большим числом замечаний."""
    if draft is None:
        return 0, 0, 0
    unsafe, problems = draft.rank
    return 1, -unsafe, -problems


def _fallback(requests: list[RequestData], citation: Citation) -> list[Answer]:
    return [
        {
            "record_id": r["record_id"],
            "text": FALLBACK_TEXTS.get(r["expert_id"], FALLBACK_TEXTS["clinician"]),
            "refused": True,
            "citations": [citation],
        }
        for r in requests
    ]


def _answers(triple: Triple, default_citation: Citation) -> list[Answer]:
    """Ответы тройки в формате /run. Нет попытки или текст небезопасен — запасной отказ."""
    draft = triple.draft
    if draft is None:
        return _fallback(triple.requests, _citation(triple.hits[0]) if triple.hits else default_citation)
    citation = _citation(triple.hits[draft.source - 1])  # ссылка у тройки одна, в том числе у запасного текста
    answers: list[Answer] = []
    for request in triple.requests:
        text = draft.texts.get(request["expert_id"])
        if isinstance(text, str) and is_safe_text(text):
            answers.append(
                {"record_id": request["record_id"], "text": tidy(text), "refused": draft.refused, "citations": [citation]}
            )
        else:
            answers += _fallback([request], citation)
    return answers


async def _prepare(
    llm: JsonModel, dataset: Dataset, index: Index, history_messages: int, cache: ResponseCache | None
) -> list[Triple]:
    """План поиска одним-двумя вызовами на набор, затем поиск: для каждой тройки — разделы и вход нейросети."""
    groups = _group_requests(dataset["requests"])
    histories = _histories(dataset.get("chat_history", []), groups, history_messages)
    try:
        plans = await _search_plans(llm, groups, histories, dataset["articles"], cache)
    except Exception:  # неожиданный ответ нейросети на этапе поиска не должен стоить всего прогона
        log.exception("план поиска не получен, тройки получат запасные ответы")
        plans = [SearchPlan() for _ in groups]
    triples = []
    for group, history, plan in zip(groups, histories, plans, strict=True):
        hits = index.search(plan.terms, k=CANDIDATES, prefer=plan.articles) if plan.terms else []
        payload = {"history": history, "utterance": group[0]["utterance"]["text"],
                   "sources": _sources(index, hits, plan.terms)}
        triples.append(Triple(group, hits, payload))
    return triples


async def answer_dataset(
    dataset: Dataset, llm: JsonModel | None = None, deadline_seconds: float | None = None, quick: bool = False,
    history_messages: int = HISTORY_MESSAGES, index: Index | None = None, cache: ResponseCache | None = None,
) -> RunResult:
    """Возвращает тело ответа /run: ровно один ответ на каждый запрос набора.

    Разговор в клиенте идёт через эту же функцию с набором из одной тройки и коротким сроком: так живой
    ответ строится теми же шагами и проходит те же проверки, что и ответы, которые оценивает стенд.
    quick — режим разговора: человек ждёт перед экраном, поэтому тройка переписывается, только если
    ответа нет или к тексту есть замечание по существу (небезопасное, текст противоречит решению об
    отказе, указание в отказе, названная причина жалоб); замечания к длине и подаче второго вызова не стоят.
    history_messages — сколько последних реплик переписки видит нейросеть: в разговоре их больше, чем в наборе.
    index — готовый индекс этой же базы статей: разговор строит его один раз на набор, а не на каждый вопрос.
    cache — память пригодных ответов нейросети (app/cache.py). Без неё каждый запрос уходит нейросети, как
    раньше; с ней повторный прогон того же набора платит только за тройки, которым ответа не хватило.
    """
    llm = llm or LLM()
    llm.deadline = time.monotonic() + (deadline_seconds or config.RUN_DEADLINE_SECONDS)
    index = index or Index(dataset["articles"])
    # Ссылка на случай, когда поиск ничего не дал: первый существующий раздел базы.
    default_citation = next(
        Citation(article_id=a["article_id"], section_id=a["sections"][0]["section_id"])
        for a in dataset["articles"] if a.get("sections")
    )
    triples = await _prepare(llm, dataset, index, history_messages, cache)
    system = prompts.answer_system(dataset["experts"])

    async def attempt(number: int, triple: Triple) -> Draft | None:
        if not triple.hits:
            return None
        try:
            return await _attempt(llm, system, triple, cache)
        except Exception:  # одна сломанная тройка не должна уронить весь прогон
            log.exception("тройка %d: попытка не удалась", number)
            return None

    # Этап 1: первая попытка для каждой тройки. Этап 2: повторы — только после того, как первый ответ
    # есть у всех, по одному и пока хватает токенов и времени. Так при нехватке суточного лимита
    # страдает качество отдельных троек, а не наличие ответов.
    first_attempts = await asyncio.gather(*(attempt(n, t) for n, t in enumerate(triples)))
    for triple, draft in zip(triples, first_attempts, strict=True):
        triple.draft = draft
    to_rewrite = sorted(
        (n for n, t in enumerate(triples) if t.hits and _needs_rewrite(t.draft, quick)),
        key=lambda n: _rewrite_priority(triples[n].draft),
    )
    rewritten = 0
    for number in to_rewrite:
        if llm.usage.total_tokens >= config.RETRY_TOKEN_BUDGET or time.monotonic() >= llm.deadline or llm.exhausted:
            break
        triple = triples[number]
        if triple.draft is not None:
            triple.payload["note"] = (
                REWRITE_NOTE + "; ".join(dict.fromkeys(triple.draft.problems))[:REWRITE_NOTE_PROBLEMS_CHARS]
            )
        triple.draft = _better(triple.draft, await attempt(number, triple))
        rewritten += 1

    by_record = {answer["record_id"]: answer for triple in triples for answer in _answers(triple, default_citation)}
    usage = llm.usage
    log.info(
        "готово: вызовов %d, токенов %d (скрытых %d), сбоев %d, повторных попыток %d из %d намеченных",
        usage.calls, usage.total_tokens, usage.reasoning_tokens, usage.failures, rewritten, len(to_rewrite),
    )
    return {"requests": [by_record[r["record_id"]] for r in dataset["requests"]]}
