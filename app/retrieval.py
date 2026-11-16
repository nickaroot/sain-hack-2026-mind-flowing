"""Лексический поиск BM25 по разделам статей. Без моделей и внешних пакетов.

Статьи приходят вместе с набором, поэтому индекс строится при каждом прогоне
(609 разделов индексируются за доли секунды). Запросы приходят уже на
английском: их готовит нейросеть одним вызовом на весь набор.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass

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
ARTICLE_DEPTH = 5  # сколько лучших разделов статьи складывается в её оценку

# Вес раздела по его типу: выводы исследования чаще лежат в аннотации, результатах и обсуждении,
# а введение и методы пересказывают чужие работы и процедуру.
_SECTION_WEIGHTS = (
    (("abstract", "conclusion", "summary", "recommendation", "practical"), 1.25),
    (("result", "discussion", "finding"), 1.15),
    (("introduction", "background", "method", "material", "statistic", "protocol", "participant"), 0.8),
    (("reference", "acknowledg", "funding", "conflict", "author", "supplement", "abbreviation"), 0.3),
)
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
    def __init__(self, articles: list[dict]):
        self.sections: list[Section] = [
            Section(a["article_id"], s["section_id"], a["title"], s["title"], s["text"])
            for a in articles
            for s in a["sections"]
        ]
        self._tf: list[Counter] = []
        df: Counter = Counter()
        for section in self.sections:
            tokens = tokenize(section.text) + tokenize(section.title) * TITLE_WEIGHT + tokenize(section.article_title) * ARTICLE_TITLE_WEIGHT
            tf = Counter(tokens)
            self._tf.append(tf)
            df.update(tf.keys())
        n = len(self.sections)
        self._weight = [section_weight(section.title) for section in self.sections]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg_len = (sum(self._len) / n) if n else 1.0
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def abstract_of(self, article_id: str) -> Section | None:
        """Аннотация статьи, а если её нет — первый раздел."""
        own = [s for s in self.sections if s.article_id == article_id]
        return next((s for s in own if "abstract" in s.title.lower()), own[0] if own else None)

    def focus(self, text: str, query: str, max_chars: int) -> str:
        """Вырезка из раздела: предложения, которые сильнее всего совпадают с запросом, в исходном порядке."""
        if len(text) <= max_chars:
            return text
        terms = set(tokenize(query))
        sentences = _SENTENCE.split(text)
        ranked = sorted(
            range(len(sentences)),
            key=lambda i: -sum(self._idf.get(t, 0.0) for t in set(tokenize(sentences[i])) & terms),
        )
        chosen, used = set(), 0
        for i in ranked:
            if used + len(sentences[i]) > max_chars and chosen:
                continue
            chosen.add(i)
            used += len(sentences[i]) + 1
            if used >= max_chars:
                break
        return " ".join(sentences[i] for i in sorted(chosen))[: max_chars + 200]

    def search(self, query: str, k: int = 6, deep: int = 2) -> list[Hit]:
        """До k разделов в порядке убывания уверенности: по одному из лучших статей и по второму из deep первых."""
        terms = Counter(tokenize(query))
        scored: list[Hit] = []
        for i, tf in enumerate(self._tf):
            score = 0.0
            norm = K1 * (1 - B + B * self._len[i] / self._avg_len)
            for term in terms:
                f = tf.get(term)
                if f:
                    score += self._idf[term] * f * (K1 + 1) / (f + norm)
            if score > 0:
                scored.append(Hit(self.sections[i], score * self._weight[i]))
        scored.sort(key=lambda h: -h.score)
        # Сначала ранжируем статьи по сумме их лучших разделов, затем берём разделы по очереди из лучших статей:
        # один случайно совпавший абзац общего обзора не должен обгонять профильную статью.
        by_article: dict[str, list[Hit]] = {}
        for hit in scored:
            by_article.setdefault(hit.section.article_id, []).append(hit)
        ranked = sorted(by_article.values(), key=lambda hits: -sum(h.score for h in hits[:ARTICLE_DEPTH]))
        picked = [hits[0] for hits in ranked[: k - deep]]
        picked += [hits[1] for hits in ranked[:deep] if len(hits) > 1]
        order = {hits[0].section.article_id: rank for rank, hits in enumerate(ranked)}
        picked.sort(key=lambda h: order[h.section.article_id])
        return picked
