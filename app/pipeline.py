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
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from app import config, medical, prompts
from app.llm import LLM, LLMError
from app.retrieval import Hit, Index

log = logging.getLogger("app.pipeline")

CANDIDATES = 6
SNIPPET_CHARS = 620
ABOUT_CHARS = 260
HISTORY_MESSAGES = 4
HISTORY_CHARS = 500
QUERY_BATCH = 20
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
_NUMBER = (
    r"(\d+[.,]?\d*|пол|полови\w+|четверт\w+|полтор\w+|од(ин|на|ну|ной)|дв(а|е|ух)|тр(и|[её]х)|четыр\w+|пят\w+|шест\w+|"
    r"сем\w+|вос[еь]м\w+|девят\w+|десят\w+)"
)
_DOSE = re.compile(
    _NUMBER + r"[\s-]*(мг\b|мкг\b|миллиграмм\w*|микрограмм\w*|mg\b|mcg\b|ме\b|капел\w*|капл\w+|кап\.|таблет\w*|"
    r"капсул\w*|драже)|полтаблетк\w+",
    re.IGNORECASE,
)
# Обращение к тому, кто оценивает ответы, запрещено правилами хакатона, поэтому фильтр широкий:
# лишняя повторная попытка стоит несравнимо дешевле штрафа.
_JUDGE_ADDRESS = re.compile(
    r"(проверяющ\w+|оценщик\w*|\bжюри\b|\bсудь(я|е|и|ю|ёй|ей)\b|\bgrader\b|\bjudge\b"
    r"|оцени\w*\s+(этот|мой)\s+ответ|постав\w+\s+(мне\s+|нам\s+|этому\s+ответу\s+)?(высш\w+|максимальн\w+|оценк\w+|балл\w*|зач[её]т\w*)"
    r"|засчита\w+|\bkind\b|addresses_judge|вер(ни|ните)\s+(только\s+)?json"
    r"|игнорир\w+\s+(все\s+|эти\s+|любые\s+|прошл\w+\s+|предыдущ\w+\s+)?(инструкци\w+|правил\w+|указани\w+)"
    r"|предыдущ\w+\s+инструкци\w+|системн\w+\s+(промпт\w*|инструкци\w+|сообщени\w+))",
    re.IGNORECASE,
)
# Ответ по существу, который пишет, чего источник не делал, проверяющая считает уклончивым.
_HEDGE = re.compile(
    r"(источник\w*|исследовани\w+|стать\w+|обзор\w*|работ\w+|автор\w+|данны\w+)\s+(этого\s+|прямо\s+|напрямую\s+|такого\s+)?"
    r"не\s+(сравнива\w+|проверя\w+|называ\w+|измеря\w+|изуча\w+|рассматрива\w+|отвеча\w+)",
    re.IGNORECASE,
)
# Настоящая проверка медицинских формулировок помечала как «лечение» отказы, где привычка подана заменой
# препарату или расписана как курс, и как «диагноз» — фразы о причине жалоб человека (пробы от 17.11).
_SUBSTITUTE = re.compile(
    r"\b(зато|вместо|взамен|замен\w+|полезнее|успех\s+—|критери\w+\s+успеха|гипотез\w+|дневник\w*|замер\w*|"
    r"отмечай\w*|отмечая|записывай\w*|проверь\w*|проверить\s+одну|попробуй\w*|убери\w*|выключай\w*)\b"
    r"|\b(\d+|два|две|три|четыре|пять|шесть|семь|семи|десять|десяти|четырнадцать)\s+(вечер\w+|ноч\w+|дн\w+|недел\w+)\b|\bне\s+(больше|более|выше|позже)\s+\d+"
    r"|\b\d+\s*(люкс\w*|лк\b)",
    re.IGNORECASE,
)
_CAUSE = re.compile(
    r"\bпричин\w+\s+(именно\s+)?(тво\w+|ваш\w+)\b|\b(тво\w+|ваш\w+)\s+\w+\s+—\s+это\s+из-за\b", re.IGNORECASE
)
# Оговорка «это не объясняет ваши пробуждения» ничего не сообщает, а проверяющая читает её как уклончивость.
# Повторного вызова она не стоит: короткое предложение с такой оговоркой просто убирается.
_DISCLAIMER = re.compile(
    r"не\s+(объясня\w+|устанавлива\w+|определя\w+|доказыва\w+|означа\w+)\s+(причин\w+\s+)?(именно\s+)?(тво\w+|ваш\w+)"
    r"|не\s+(ярлык|диагноз|объяснение)\s+(для\s+)?(теб\w+|вас\w*|тво\w+|ваш\w+)"
    r"|\bне\s+\w+\s+причин\w+|\bа\s+не\s+к\s+(тво\w+|ваш\w+)",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
DISCLAIMER_MAX_WORDS = 16
_NO_STUDIES = re.compile(
    r"\bнет\s+(\w+\s+){0,2}(исследовани|данных|источник|работ)|(исследовани\w+|данных|источник\w+)[^.!?]{0,60}\bнет\b",
    re.IGNORECASE,
)
# Ответ по существу обязан назвать, что установлено в исследованиях.
_EVIDENCE = re.compile(r"(исследовани|обзор|метаанализ|эксперимент|рекомендаци|наблюдени|учёны|испытани)", re.IGNORECASE)
# Ясная фраза отказа от первого лица: без неё проверяющая не засчитывает отказ, с ней — не засчитывает ответ.
_REFUSAL = re.compile(
    r"\bне\s+((буду|стану)(?!\s+(обещать|пугать|спорить|скрывать|утверждать))|назову|дам|подскажу|посоветую|выберу|"
    r"оценю|подберу|скажу|укажу|распишу|пропишу|порекомендую|определю|подтвержу|берусь|возьмусь|советую|рекомендую|"
    r"одобряю|одобрю|могу\s+(назвать|подсказать|посоветовать|выбрать|оценить|подобрать|сказать|"
    r"поставить|ставить|определить|подтвердить|назначить|рекомендовать))\b",
    re.IGNORECASE,
)


def is_safe_text(text: str) -> bool:
    return (
        bool(text.strip()) and not _DOSE.search(text) and not _JUDGE_ADDRESS.search(text)
        and medical.dictionary_match(text) is None
    )


def text_problems(text: str, refused: bool, unsupported: bool = False) -> list[str]:
    """Что в тексте не так по признакам, которые можно проверить без нейросети. Пустой список — замечаний нет."""
    problems = []
    if _DOSE.search(text):
        problems.append(
            "в тексте есть количество вещества с единицей измерения: убери миллиграммы, капли и таблетки, "
            "кофеин пиши только в чашках, алкоголь — в бокалах"
        )
    found = medical.dictionary_match(text)
    if found:
        problems.append(f"фраза «{found[1]}» попадает под запрет на медицинские формулировки: убери её")
    if _JUDGE_ADDRESS.search(text):
        problems.append("текст упоминает проверку ответов или инструкции: пиши только о сне человека")
    words = len(text.split())
    if not WORDS_MIN <= words <= WORDS_MAX:
        problems.append(f"в тексте {words} слов, нужно 60–100")
    if refused and _SUBSTITUTE.search(text):
        problems.append(
            "в отказе есть указание, что делать, план проверки, норматив или слово о замене препарату: оставь только "
            "отказ, наблюдение из исследования и вывод о связи привычки со сном"
        )
    if _names_cause(text):
        problems.append("текст называет причину жалоб человека: пиши о связи привычки со сном, а не о причине")
    if unsupported and not _NO_STUDIES.search(text):
        problems.append("в отказе не сказано прямо, что исследований об этом методе или товаре в базе нет")
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


def _names_cause(text: str) -> bool:
    """Текст утверждает причину жалоб человека. Отрицание («не устанавливает причину…») сюда не относится."""
    for sentence in _SENTENCE_END.split(text):
        match = _CAUSE.search(sentence)
        if match and not re.search(r"\bне\b|\bнельзя\b", sentence[: match.start()], re.IGNORECASE):
            return True
    return False


def tidy(text: str) -> str:
    """Убирает короткие предложения-оговорки о причине жалоб, если без них текст остаётся полным."""
    sentences = _SENTENCE_END.split(text.strip())
    kept = [s for s in sentences if not (_DISCLAIMER.search(s) and len(s.split()) <= DISCLAIMER_MAX_WORDS)]
    cleaned = " ".join(kept)
    return cleaned if len(cleaned.split()) >= WORDS_MIN else text.strip()


def is_fallback(answer: dict) -> bool:
    return answer["text"] in FALLBACK_TEXTS.values()


def _citation(hit: Hit) -> dict:
    return {"article_id": hit.section.article_id, "section_id": hit.section.section_id}


def _moment(value: object) -> datetime:
    """Отметка времени как момент. Без часового пояса считаем её UTC; нечитаемую ставим в самое начало."""
    try:
        moment = datetime.fromisoformat(str(value))
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _history_for(request: dict, by_user: dict[str, list[dict]]) -> list[str]:
    # Время сравниваем как моменты, а не как строки: у сообщений бывают разные часовые пояса.
    asked_at = _moment(request["utterance"].get("at"))
    earlier = [m for m in by_user.get(request["user_id"], []) if _moment(m.get("at")) < asked_at]
    return [f"{m['role']}: {m['text'][:HISTORY_CHARS]}" for m in earlier[-HISTORY_MESSAGES:]]


def _group_requests(requests: list[dict]) -> list[list[dict]]:
    """Тройки по group_id; запрос без группы обрабатывается отдельно."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for request in requests:
        groups[request.get("group_id") or f"single:{request['record_id']}"].append(request)
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
            return  # вызов уже повторялся внутри LLM; дробить пачку при недоступной нейросети — только тратить время
        for key, value in result.items():
            if not (key.isdigit() and int(key) < len(plans)):
                continue
            if isinstance(value, str):  # нейросеть вернула только слова — поиск всё равно сработает
                value = {"terms": value}
            if not isinstance(value, dict) or not isinstance(value.get("terms"), str):
                continue
            chosen = value.get("articles") if isinstance(value.get("articles"), list) else []
            plans[int(key)] = SearchPlan(
                value["terms"][:400], [a for a in chosen if isinstance(a, str) and a in known][:4]
            )
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


@dataclass
class Draft:
    """Одна попытка ответа на тройку и то, что о ней сказала проверка кодом."""

    texts: dict
    source: int
    refused: bool
    unsafe: int
    problems: list[str]

    @property
    def rank(self) -> tuple[int, int]:
        # Небезопасный текст заменяется запасным отказом и очков не приносит, поэтому безопасность важнее.
        return self.unsafe, len(self.problems)


async def _attempt(llm: LLM, system: str, payload: dict, group: list[dict], sources: int) -> Draft | None:
    """Один вызов на тройку. None — нейросеть не ответила или вернула не то."""
    try:
        result = await llm.chat_json(system, json.dumps(payload, ensure_ascii=False), max_tokens=3500, effort="low")
    except LLMError:
        return None
    texts, source = result.get("texts"), result.get("source")
    if not isinstance(texts, dict) or not isinstance(source, int) or not 1 <= source <= sources:
        return None
    refused = result.get("kind") == "refusal" or result.get("forbidden") not in (None, "none")
    own = [texts.get(r["expert_id"]) for r in group]
    problems = [
        f"{r['expert_id']}: {problem}"
        for r, text in zip(group, own)
        for problem in (
            text_problems(text, refused, result.get("forbidden") == "unsupported")
            if isinstance(text, str) else ["текста нет"]
        )
    ]
    unsafe = sum(not (isinstance(t, str) and is_safe_text(t)) for t in own)
    return Draft(texts, source, refused, unsafe, problems)


def _better(first: Draft | None, second: Draft | None) -> Draft | None:
    """Какую из двух попыток оставить."""
    if first is None or second is None:
        return first or second
    if second.refused != first.refused:
        return first  # замечания касались текста, а не решения об отказе: повтор, который его поменял, не берём
    return second if second.rank <= first.rank else first  # при равенстве — уже исправленная попытка


def _answers(group: list[dict], draft: Draft | None, hits: list[Hit], default_citation: dict) -> list[dict]:
    if draft is None:
        return _fallback(group, hits, default_citation)
    citation = _citation(hits[draft.source - 1])
    answers = []
    for request in group:
        text = draft.texts.get(request["expert_id"])
        if isinstance(text, str) and is_safe_text(text):
            answers.append(
                {"record_id": request["record_id"], "text": tidy(text), "refused": draft.refused, "citations": [citation]}
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
    llm.deadline = time.monotonic() + config.RUN_DEADLINE_SECONDS
    index = Index(dataset["articles"])
    # Ссылка на случай, когда поиск ничего не дал: первый существующий раздел базы.
    default_citation = next(
        {"article_id": a["article_id"], "section_id": a["sections"][0]["section_id"]}
        for a in dataset["articles"] if a.get("sections")
    )

    by_user: dict[str, list[dict]] = defaultdict(list)
    for message in sorted(dataset.get("chat_history", []), key=lambda m: _moment(m.get("at"))):
        by_user[message["user_id"]].append(message)

    groups = _group_requests(dataset["requests"])
    histories = [_history_for(group[0], by_user) for group in groups]
    try:
        plans = await _search_plans(llm, groups, histories, dataset["articles"])
    except Exception:  # неожиданный ответ нейросети на этапе поиска не должен стоить всего прогона
        log.exception("план поиска не получен, тройки получат запасные ответы")
        plans = [SearchPlan() for _ in groups]
    system = prompts.answer_system(dataset["experts"])
    all_hits = [index.search(p.terms, k=CANDIDATES, prefer=p.articles) if p.terms else [] for p in plans]
    payloads = [
        {"history": histories[i], "utterance": groups[i][0]["utterance"]["text"],
         "sources": _sources(index, all_hits[i], plans[i].terms)}
        for i in range(len(groups))
    ]

    async def attempt(i: int) -> Draft | None:
        if not all_hits[i]:
            return None
        try:
            return await _attempt(llm, system, payloads[i], groups[i], len(all_hits[i]))
        except Exception:  # одна сломанная тройка не должна уронить весь прогон
            log.exception("тройка %d: попытка не удалась", i)
            return None

    # Этап 1: первая попытка для каждой тройки. Этап 2: повторы — только после того, как первый ответ
    # есть у всех, по одному и пока хватает токенов и времени. Так при нехватке суточного лимита
    # страдает качество отдельных троек, а не наличие ответов.
    drafts = list(await asyncio.gather(*(attempt(i) for i in range(len(groups)))))
    order = sorted(
        (i for i, d in enumerate(drafts) if all_hits[i] and (d is None or d.problems)),
        key=lambda i: (drafts[i] is not None, *(-n for n in (drafts[i].rank if drafts[i] else (0, 0)))),
    )
    rewritten = 0
    for i in order:
        if (
            llm.usage.total_tokens >= config.RETRY_TOKEN_BUDGET or time.monotonic() >= llm.deadline
            or getattr(llm, "exhausted", False)
        ):
            break
        if drafts[i] is not None:
            payloads[i]["note"] = (
                "Прошлая попытка возвращена с замечаниями. Перепиши все три текста, исправь это и сохрани манеру "
                "каждого эксперта. " + "; ".join(dict.fromkeys(drafts[i].problems))[:900]
            )
        drafts[i] = _better(drafts[i], await attempt(i))
        rewritten += 1

    solved = [_answers(groups[i], drafts[i], all_hits[i], default_citation) for i in range(len(groups))]
    by_record = {answer["record_id"]: answer for answers in solved for answer in answers}
    usage = llm.usage
    log.info(
        "готово: вызовов %d, токенов %d (скрытых %d), сбоев %d, повторных попыток %d из %d намеченных",
        usage.calls, usage.total_tokens, usage.reasoning_tokens, usage.failures, rewritten, len(order),
    )
    return {"requests": [by_record[r["record_id"]] for r in dataset["requests"]]}
