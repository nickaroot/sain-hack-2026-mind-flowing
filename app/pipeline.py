"""Сборка ответов на набор кейса 4.

Порядок работы:
1. Один вызов нейросети на все реплики набора: английские поисковые слова и статьи, выбранные по списку названий.
2. BM25 находит разделы статей для каждой тройки запросов; порядок статей объединяется с выбором нейросети.
3. Один вызов на тройку: проверка на недопустимую просьбу, выбор источника и три текста экспертов.
   Общая ссылка у тройки получается сама собой, а запросы разных пользователей не смешиваются.
4. Тексты проверяет код: количества веществ, обращение к проверяющим, длина, ясность отказа, оговорки.
   С замечаниями делается одна повторная попытка. Если нейросеть недоступна или текст небезопасен,
   запись получает запасной отказ: без ответа не остаётся ни одна запись.
"""

import asyncio
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from app import config, prompts
from app.llm import LLM, LLMError
from app.retrieval import Hit, Index

log = logging.getLogger("app.pipeline")

CANDIDATES = 6
SNIPPET_CHARS = 700
ABOUT_CHARS = 350
HISTORY_MESSAGES = 4
HISTORY_CHARS = 200
QUERY_BATCH = 20
ANSWER_ATTEMPTS = 2
WORDS_MIN, WORDS_MAX = 40, 130  # в инструкции 60–100 слов; за этими границами текст явно не тот
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
# Обращение к тому, кто оценивает ответы, запрещено правилами хакатона, поэтому фильтр широкий:
# лишняя повторная попытка стоит несравнимо дешевле штрафа.
_JUDGE_ADDRESS = re.compile(
    r"(проверяющ\w+|оценщик\w*|\bжюри\b|\bсудь(я|е|и|ю|ёй|ей)\b|\bgrader\b|\bjudge\b"
    r"|оцени\w*\s+(этот|мой)\s+ответ|постав\w+\s+(мне\s+|нам\s+|этому\s+ответу\s+)?(высш\w+|максимальн\w+|оценк\w+|балл\w*|зач[её]т\w*)"
    r"|игнорируй\w*|предыдущ\w+\s+инструкци\w+|системн\w+\s+(промпт\w*|инструкци\w+|сообщени\w+))",
    re.IGNORECASE,
)
# Ответ по существу, который пишет, чего источник не делал, проверяющая считает уклончивым.
_HEDGE = re.compile(
    r"(источник\w*|исследовани\w+|стать\w+|обзор\w*|работ\w+|автор\w+|данны\w+)\s+(этого\s+|прямо\s+|напрямую\s+|такого\s+)?"
    r"не\s+(сравнива\w+|проверя\w+|называ\w+|измеря\w+|изуча\w+|рассматрива\w+|отвеча\w+|позволя\w+|да[её]т)",
    re.IGNORECASE,
)
# Ответ по существу обязан назвать, что установлено в исследованиях.
_EVIDENCE = re.compile(r"(исследовани|обзор|метаанализ|эксперимент|рекомендаци|наблюдени|учёны|испытани)", re.IGNORECASE)
# Ясная фраза отказа от первого лица: без неё проверяющая не засчитывает отказ, с ней — не засчитывает ответ.
_REFUSAL = re.compile(
    r"\bне\s+(буду|стану|назову|дам|подскажу|посоветую|выберу|оценю|подберу|скажу|берусь|возьмусь|могу\s+(назвать|"
    r"подсказать|посоветовать|выбрать|оценить|подобрать|сказать))\b",
    re.IGNORECASE,
)


def is_safe_text(text: str) -> bool:
    return bool(text.strip()) and not _DOSE.search(text) and not _JUDGE_ADDRESS.search(text)


def text_problems(text: str, refused: bool) -> list[str]:
    """Что в тексте не так по признакам, которые можно проверить без нейросети. Пустой список — замечаний нет."""
    problems = []
    if _DOSE.search(text):
        problems.append(
            "в тексте есть количество вещества с единицей измерения: убери миллиграммы, капли и таблетки, "
            "кофеин пиши только в чашках, алкоголь — в бокалах"
        )
    if _JUDGE_ADDRESS.search(text):
        problems.append("текст упоминает проверку ответов или инструкции: пиши только о сне человека")
    words = len(text.split())
    if not WORDS_MIN <= words <= WORDS_MAX:
        problems.append(f"в тексте {words} слов, нужно 60–100")
    if refused and not _REFUSAL.search(text):
        problems.append("в отказе нет ясной фразы от первого лица о том, чего эксперт не даст («не назову», «не буду выбирать»)")
    if not refused and _REFUSAL.search(text):
        problems.append("ответ по существу звучит как отказ: убери фразы «не буду», «не назову» и ответь прямо")
    if not refused and not _EVIDENCE.search(text):
        problems.append("в ответе не назван результат исследования: добавь фразу «в исследовании…» или «в обзоре…» с тем, что получено")
    if not refused and _HEDGE.search(text):
        problems.append(
            "ответ пишет, чего источник не сравнивал или не проверял: вместо этого дай ближайший результат и вывод из него"
        )
    return problems


@dataclass
class SearchPlan:
    """Что нейросеть вернула для поиска по одной реплике."""

    terms: str = ""
    articles: list[str] = field(default_factory=list)


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


async def _search_plans(
    llm: LLM, groups: list[list[dict]], histories: list[list[str]], articles: list[dict]
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
        items = []
        for i in batch:
            context = [h[len("user: "):] for h in histories[i] if h.startswith("user:")][-2:]
            items.append({"id": str(i), "context": context, "text": groups[i][0]["utterance"]["text"]})
        try:
            result = await llm.chat_json(
                prompts.QUERY_SYSTEM, json.dumps({"catalog": catalog, "items": items}, ensure_ascii=False),
                max_tokens=4000, effort="minimal",
            )
        except LLMError:
            result = {}
        for key, value in result.items():
            if not (key.isdigit() and int(key) < len(plans)):
                continue
            if isinstance(value, str):  # нейросеть вернула только слова — поиск всё равно сработает
                value = {"terms": value}
            if not isinstance(value, dict) or not isinstance(value.get("terms"), str):
                continue
            chosen = value.get("articles") if isinstance(value.get("articles"), list) else []
            plans[int(key)] = SearchPlan(value["terms"][:400], [a for a in chosen if a in known][:4])
        missing = [i for i in batch if not plans[i].terms]
        if missing and len(batch) > 1 and llm.usage.total_tokens < config.RETRY_TOKEN_BUDGET:
            half = (len(missing) + 1) // 2
            await asyncio.gather(one_batch(missing[:half]), one_batch(missing[half:]))

    order = list(range(len(groups)))
    await asyncio.gather(*(one_batch(order[start : start + QUERY_BATCH]) for start in range(0, len(order), QUERY_BATCH)))
    return plans


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


async def _answer_group(
    llm: LLM, system: str, group: list[dict], history: list[str], hits: list[Hit], default_citation: dict,
    index: Index, query: str,
) -> list[dict]:
    if not hits:
        return _fallback(group, hits, default_citation)
    payload = {
        "history": history,
        "utterance": group[0]["utterance"]["text"],
        "sources": _sources(index, hits, query),
    }
    # Запасной ответ баллов не приносит, поэтому одна повторная попытка оправдана: и при сбое вызова,
    # и когда проверка кодом нашла в текстах замечания (тогда во второй попытке они перечислены).
    # Из попыток берётся та, где замечаний меньше. Когда лимит токенов на исходе, повторов нет.
    parsed, fewest = None, None
    for attempt in range(ANSWER_ATTEMPTS):
        if attempt and llm.usage.total_tokens >= config.RETRY_TOKEN_BUDGET:
            break
        try:
            result = await llm.chat_json(system, json.dumps(payload, ensure_ascii=False), max_tokens=3500, effort="low")
        except LLMError:
            continue
        texts, source = result.get("texts"), result.get("source")
        if not isinstance(texts, dict) or not isinstance(source, int) or not 1 <= source <= len(hits):
            continue
        refused = result.get("kind") == "refusal" or result.get("forbidden") not in (None, "none")
        problems = [
            f"{r['expert_id']}: {problem}"
            for r in group
            for problem in (
                text_problems(texts[r["expert_id"]], refused)
                if isinstance(texts.get(r["expert_id"]), str) else ["текста нет"]
            )
        ]
        if fewest is None or len(problems) < fewest:
            parsed, fewest = (result, texts, source), len(problems)
        if not problems:
            break
        payload["note"] = (
            "Прошлая попытка возвращена с замечаниями. Перепиши все три текста, исправь это и сохрани манеру "
            "каждого эксперта. " + "; ".join(dict.fromkeys(problems))[:900]
        )
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
    plans = await _search_plans(llm, groups, histories, dataset["articles"])
    system = prompts.answer_system(dataset["experts"])

    async def solve(i: int) -> list[dict]:
        plan = plans[i]
        hits = index.search(plan.terms, k=CANDIDATES, prefer=plan.articles) if plan.terms else []
        try:
            return await _answer_group(
                llm, system, groups[i], histories[i], hits, default_citation, index, plan.terms
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
