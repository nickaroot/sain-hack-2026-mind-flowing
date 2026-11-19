"""Лексический поиск BM25 по разделам статей. Без моделей и внешних пакетов.

Статьи приходят вместе с набором, поэтому индекс строится при каждом прогоне
(609 разделов индексируются за доли секунды). Запросы приходят уже на
английском: их готовит нейросеть одним вызовом на весь набор.

Индекс обратный: для каждого слова хранится список разделов, где оно встречается. Запрос просматривает
только разделы со своими словами, поэтому время поиска растёт с числом совпадений, а не с размером базы.
"""

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from app.contracts import ArticleData

_WORD = re.compile(r"[a-z][a-z0-9-]+")
_STOP = frozenset(
    "the of and in to a is was were are for with on by as at that this these those from be been or "
    "an it its their they we our which than but not no can may also both between into during after "
    "before more most less such has have had using used use study studies results participants "
    "however there all one two three other each".split()
)
_SUFFIXES = ("ization", "ations", "ation", "ingly", "ness", "ment", "ings", "ing", "ies", "ied", "ed", "es", "ly", "s")

K1, B = 1.4, 0.7
TITLE_WEIGHT = 2  # заголовок раздела учитывается дважды
ARTICLE_TITLE_WEIGHT = 3  # название статьи задаёт её тему: помогает отличить профильную статью от общего обзора
FUSION_C = 2  # маленькая константа: первые места в обоих порядках весят заметно больше остальных
ARTICLE_DEPTH = 5  # сколько лучших разделов статьи складывается в её оценку

# Вес раздела по его типу: выводы исследования чаще лежат в аннотации, результатах и обсуждении,
# а введение и методы пересказывают чужие работы и процедуру.
_SECTION_WEIGHTS = (
    (("abstract", "conclusion", "summary", "recommendation", "practical"), 1.25),
    (("result", "discussion", "finding"), 1.15),
    (("introduction", "background", "method", "material", "statistic", "protocol", "participant"), 0.8),
    (("reference", "acknowledg", "funding", "conflict", "author", "supplement", "abbreviation"), 0.3),
)
_DIGIT = re.compile(r"\d")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")


def section_weight(title: str) -> float:
    lowered = title.lower()
    for markers, weight in _SECTION_WEIGHTS:
        if any(marker in lowered for marker in markers):
            return weight
    return 1.0


def _stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    return [_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP]


@dataclass(frozen=True)
class Section:
    article_id: str
    section_id: str
    article_title: str
    title: str
    text: str


@dataclass(frozen=True)
class Hit:
    section: Section
    score: float


class Index:
    def __init__(self, articles: Sequence[ArticleData]):
        self.sections: list[Section] = [
            Section(a["article_id"], s["section_id"], a["title"], s["title"], s["text"])
            for a in articles
            for s in a["sections"]
        ]
        # Слово → разделы, где оно есть, с числом вхождений; разделы идут по возрастанию номера.
        self._postings: dict[str, list[tuple[int, int]]] = {}
        lengths: list[int] = []
        self._abstracts: dict[str, Section] = {}
        first_sections: dict[str, Section] = {}
        for number, section in enumerate(self.sections):
            tokens = (
                tokenize(section.text)
                + tokenize(section.title) * TITLE_WEIGHT
                + tokenize(section.article_title) * ARTICLE_TITLE_WEIGHT
            )
            for term, count in Counter(tokens).items():
                self._postings.setdefault(term, []).append((number, count))
            lengths.append(len(tokens))
            first_sections.setdefault(section.article_id, section)
            if "abstract" in section.title.lower():
                self._abstracts.setdefault(section.article_id, section)
        for article_id, section in first_sections.items():  # аннотации нет — её заменяет первый раздел
            self._abstracts.setdefault(article_id, section)
        n = len(self.sections)
        avg_len = (sum(lengths) / n) if n else 1.0
        self._weight = [section_weight(section.title) for section in self.sections]
        # Поправка BM25 на длину раздела не зависит от запроса: считается один раз.
        self._norm = [K1 * (1 - B + B * length / avg_len) for length in lengths]
        self._idf = {t: math.log(1 + (n - len(found) + 0.5) / (len(found) + 0.5)) for t, found in self._postings.items()}

    def _fuse(self, ranked: list[list[Hit]], by_article: dict[str, list[Hit]], prefer: Sequence[str]) -> list[list[Hit]]:
        """Объединение двух порядков статей по обратным рангам (reciprocal rank fusion)."""
        # При равенстве выигрывает выбор по названию: он стоит в словаре первым, а сортировка устойчива.
        score: dict[str, float] = {}
        for rank, article_id in enumerate(dict.fromkeys(prefer)):
            if article_id not in by_article:
                # Слова запроса в статье не встретились, но нейросеть её выбрала: берём аннотацию.
                abstract = self.abstract_of(article_id)
                if abstract is None:
                    continue
                by_article[article_id] = [Hit(abstract, 0.0)]
            score[article_id] = 1.0 / (FUSION_C + rank + 1)
        for rank, hits in enumerate(ranked):
            article_id = hits[0].section.article_id
            score[article_id] = score.get(article_id, 0.0) + 1.0 / (FUSION_C + rank + 1)
        return [by_article[a] for a in sorted(score, key=lambda a: -score[a])]

    def abstract_of(self, article_id: str) -> Section | None:
        """Аннотация статьи, а если её нет — первый раздел."""
        return self._abstracts.get(article_id)

    def focus(self, text: str, query: str, max_chars: int) -> str:
        """Вырезка из раздела: предложения, которые сильнее всего совпадают с запросом, в исходном порядке."""
        if len(text) <= max_chars:
            return text
        terms = set(tokenize(query))
        sentences = _SENTENCE.split(text)

        def closeness(sentence: str) -> float:
            # Слова складываются по алфавиту: порядок обхода множества у каждого процесса свой, а сумма дробных
            # чисел от порядка зависит. Без этого одна и та же вырезка менялась бы от перезапуска к перезапуску.
            matched = sum(self._idf.get(term, 0.0) for term in sorted(set(tokenize(sentence)) & terms))
            # Предложение с числом — обычно измеренный результат, его и просят в ответе: небольшой перевес.
            return matched * (1.25 if _DIGIT.search(sentence) else 1.0)

        ranked = sorted(range(len(sentences)), key=lambda i: -closeness(sentences[i]))
        chosen: set[int] = set()
        used = 0
        for i in ranked:
            if used + len(sentences[i]) > max_chars and chosen:
                continue
            chosen.add(i)
            used += len(sentences[i]) + 1
            if used >= max_chars:
                break
        return " ".join(sentences[i] for i in sorted(chosen))[: max_chars + 200]

    def search(self, query: str, k: int = 6, deep: int = 2, prefer: Sequence[str] = ()) -> list[Hit]:
        """До k разделов в порядке убывания уверенности: по одному из лучших статей и по второму из deep первых.

        prefer — статьи, которые нейросеть выбрала по списку названий, от лучшей к худшей. Их порядок
        объединяется с порядком поиска по словам: название статьи понимает смысл вопроса, а слова находят раздел.
        """
        scores: dict[int, float] = {}
        for term in dict.fromkeys(tokenize(query)):  # повтор слова в запросе веса не добавляет
            idf = self._idf.get(term)
            if idf is None:
                continue
            for number, count in self._postings[term]:
                scores[number] = scores.get(number, 0.0) + idf * count * (K1 + 1) / (count + self._norm[number])
        # Разделы перебираются по номеру, а сортировка устойчива: при равной оценке раньше идёт раздел,
        # который раньше стоит в базе. От этого порядка зависит, какие разделы увидит нейросеть.
        scored = [Hit(self.sections[n], scores[n] * self._weight[n]) for n in sorted(scores) if scores[n] > 0]
        scored.sort(key=lambda h: -h.score)
        # Сначала ранжируем статьи по сумме их лучших разделов, затем берём разделы по очереди из лучших статей:
        # один случайно совпавший абзац общего обзора не должен обгонять профильную статью.
        by_article: dict[str, list[Hit]] = {}
        for hit in scored:
            by_article.setdefault(hit.section.article_id, []).append(hit)
        ranked = sorted(by_article.values(), key=lambda hits: -sum(h.score for h in hits[:ARTICLE_DEPTH]))
        if prefer:
            ranked = self._fuse(ranked, by_article, prefer)
        picked = [hits[0] for hits in ranked[: k - deep]]
        picked += [hits[1] for hits in ranked[:deep] if len(hits) > 1]
        # У лучших статей может не быть второго раздела: тогда свободные места занимают следующие статьи.
        picked += [hits[0] for hits in ranked[k - deep :]][: k - len(picked)]
        order = {hits[0].section.article_id: rank for rank, hits in enumerate(ranked)}
        picked.sort(key=lambda h: order[h.section.article_id])
        return picked
