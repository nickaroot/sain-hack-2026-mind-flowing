"""Сборка ответов на набор кейса 4.

Порядок работы:
1. Один вызов нейросети превращает все реплики набора в английские поисковые слова.
2. BM25 находит разделы статей для каждой тройки запросов.
3. Один вызов на тройку: проверка на недопустимую просьбу, выбор источника и три текста экспертов.
   Общая ссылка у тройки получается сама собой, а запросы разных пользователей не смешиваются.
4. Тексты проходят детерминированный фильтр. Если нейросеть недоступна или текст не прошёл фильтр,
   запись получает запасной отказ: без ответа не остаётся ни одна запись.
"""

import asyncio
import json
import logging
import re
from collections import defaultdict
from datetime import datetime

from app import prompts
from app.llm import LLM, LLMError
from app.retrieval import Hit, Index

log = logging.getLogger("app.pipeline")

CANDIDATES = 6
SNIPPET_CHARS = 800
HISTORY_MESSAGES = 4
HISTORY_CHARS = 200
QUERY_BATCH = 20
ANSWER_ATTEMPTS = 2
RETRY_NOTE = (
    "В прошлой попытке в тексте было количество вещества с единицей измерения. Перепиши все три текста без "
    "миллиграммов, капель и таблеток: кофеин — только в чашках, алкоголь — в бокалах."
)
EXPERT_IDS = ("clinician", "biohacker", "nutritionist")

FALLBACK_TEXTS = {
    "clinician": "На этот вопрос я не отвечу: он требует очного разговора со специалистом. "
    "Из исследований о сне можно взять один безопасный вывод — он в разделе статьи по ссылке.",
    "biohacker": "Это не дам: тут нужен врач, а не эксперимент. Что можно проверить самому — "
    "смотри раздел статьи по ссылке и выбери одну привычку для замера на неделю.",
    "nutritionist": "Здесь я не подскажу, с этим лучше к врачу. А пока загляни в раздел статьи "
    "по ссылке: там есть простой шаг, который можно попробовать уже сегодня вечером.",
}

# Количество вещества рядом с единицей измерения: такого в текстах быть не должно.
_DOSE = re.compile(
    r"\d+[.,]?\d*\s*(мг|мкг|миллиграмм\w*|микрограмм\w*|mg|mcg|ме\b|капел\w*|кап\.|таблет\w*|капсул\w*|драже)",
    re.IGNORECASE,
)
_JUDGE_ADDRESS = re.compile(r"(проверяющ\w+|оцени\w*\s+этот\s+ответ|поставь\w*\s+оценк\w+)", re.IGNORECASE)


def is_safe_text(text: str) -> bool:
    return bool(text.strip()) and not _DOSE.search(text) and not _JUDGE_ADDRESS.search(text)


def _citation(hit: Hit) -> dict:
    return {"article_id": hit.section.article_id, "section_id": hit.section.section_id}


def _history_for(request: dict, by_user: dict[str, list[dict]]) -> list[str]:
    # Время сравниваем как моменты, а не как строки: у сообщений бывают разные часовые пояса.
    asked_at = datetime.fromisoformat(request["utterance"]["at"])
    earlier = [m for m in by_user.get(request["user_id"], []) if datetime.fromisoformat(m["at"]) < asked_at]
    return [f"{m['role']}: {m['text'][:HISTORY_CHARS]}" for m in earlier[-HISTORY_MESSAGES:]]


def _group_requests(requests: list[dict]) -> list[list[dict]]:
    """Тройки по group_id; запрос без группы обрабатывается отдельно."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for request in requests:
        groups[request["group_id"] or f"single:{request['record_id']}"].append(request)
    return list(groups.values())


async def _search_queries(llm: LLM, groups: list[list[dict]], histories: list[list[str]]) -> list[str]:
    """Английские поисковые слова для каждой группы. При сбое остаётся пустая строка."""
    queries = [""] * len(groups)

    async def one_batch(start: int) -> None:
        items = []
        for i in range(start, min(start + QUERY_BATCH, len(groups))):
            context = [h[len("user: "):] for h in histories[i] if h.startswith("user:")][-2:]
            items.append({"id": str(i), "context": context, "text": groups[i][0]["utterance"]["text"]})
        try:
            result = await llm.chat_json(
                prompts.QUERY_SYSTEM, json.dumps(items, ensure_ascii=False), max_tokens=2500, effort="minimal"
            )
        except LLMError:
            return
        for key, value in result.items():
            if key.isdigit() and int(key) < len(queries) and isinstance(value, str):
                queries[int(key)] = value[:400]

    await asyncio.gather(*(one_batch(start) for start in range(0, len(groups), QUERY_BATCH)))
    return queries


def _fallback(group: list[dict], hits: list[Hit], default_citation: dict) -> list[dict]:
    citation = _citation(hits[0]) if hits else default_citation
    return [
        {
            "record_id": r["record_id"],
            "text": FALLBACK_TEXTS.get(r["expert_id"], FALLBACK_TEXTS["clinician"]),
            "refused": True,
            "citations": [citation],
        }
        for r in group
    ]


async def _answer_group(
    llm: LLM, system: str, group: list[dict], history: list[str], hits: list[Hit], default_citation: dict,
    index: Index, query: str,
) -> list[dict]:
    if not hits:
        return _fallback(group, hits, default_citation)
    payload = {
        "history": history,
        "utterance": group[0]["utterance"]["text"],
        "sources": [
            {"n": n, "article": h.section.article_title, "section": h.section.title,
             "text": index.focus(h.section.text, query, SNIPPET_CHARS)}
            for n, h in enumerate(hits, start=1)
        ],
    }
    # Запасной ответ баллов не приносит, поэтому одна повторная попытка оправдана: и при сбое вызова,
    # и когда в тексте оказалось количество вещества (тогда во второй попытке об этом сказано прямо).
    parsed = None
    for attempt in range(ANSWER_ATTEMPTS):
        try:
            result = await llm.chat_json(system, json.dumps(payload, ensure_ascii=False), max_tokens=3500, effort="low")
        except LLMError:
            continue
        texts, source = result.get("texts"), result.get("source")
        if not isinstance(texts, dict) or not isinstance(source, int) or not 1 <= source <= len(hits):
            continue
        parsed = (result, texts, source)
        if all(isinstance(texts.get(r["expert_id"]), str) and is_safe_text(texts[r["expert_id"]]) for r in group):
            break
        payload["note"] = RETRY_NOTE
    if parsed is None:
        return _fallback(group, hits, default_citation)
    result, texts, source = parsed
    refused = result.get("kind") == "refusal" or result.get("forbidden") not in (None, "none")
    citation = _citation(hits[source - 1])

    answers = []
    for request in group:
        text = texts.get(request["expert_id"])
        if isinstance(text, str) and is_safe_text(text):
            answers.append(
                {"record_id": request["record_id"], "text": text.strip(), "refused": refused, "citations": [citation]}
            )
        else:
            # Запасной текст — отказ, но ссылка остаётся общей для тройки.
            fallback = _fallback([request], hits, default_citation)[0]
            fallback["citations"] = [citation]
            answers.append(fallback)
    return answers


async def answer_dataset(dataset: dict, llm: LLM | None = None) -> dict:
    """Возвращает тело ответа /run: ровно один ответ на каждый запрос набора."""
    llm = llm or LLM()
    index = Index(dataset["articles"])
    first_article = dataset["articles"][0]
    default_citation = {
        "article_id": first_article["article_id"],
        "section_id": first_article["sections"][0]["section_id"],
    }

    by_user: dict[str, list[dict]] = defaultdict(list)
    for message in sorted(dataset.get("chat_history", []), key=lambda m: datetime.fromisoformat(m["at"])):
        by_user[message["user_id"]].append(message)

    groups = _group_requests(dataset["requests"])
    histories = [_history_for(group[0], by_user) for group in groups]
    queries = await _search_queries(llm, groups, histories)
    system = prompts.answer_system(dataset["experts"])

    async def solve(i: int) -> list[dict]:
        hits = index.search(queries[i], k=CANDIDATES) if queries[i] else []
        try:
            return await _answer_group(
                llm, system, groups[i], histories[i], hits, default_citation, index, queries[i]
            )
        except Exception:  # одна сломанная тройка не должна уронить весь прогон
            log.exception("тройка %d обработана запасным ответом", i)
            return _fallback(groups[i], hits, default_citation)

    solved = await asyncio.gather(*(solve(i) for i in range(len(groups))))
    by_record = {answer["record_id"]: answer for answers in solved for answer in answers}
    usage = llm.usage
    log.info(
        "готово: вызовов %d, токенов %d (скрытых %d), сбоев %d",
        usage.calls, usage.total_tokens, usage.reasoning_tokens, usage.failures,
    )
    return {"requests": [by_record[r["record_id"]] for r in dataset["requests"]]}
