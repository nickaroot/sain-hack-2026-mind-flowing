"""Поиск по базе статей: обратный индекс обязан находить то же, что прямой перебор всех разделов."""

import math
import random
from collections import Counter

from app import retrieval
from app.retrieval import Index, tokenize

WORDS = ("sleep caffeine melatonin light evening morning nap exercise alcohol screen latency duration "
         "insomnia bedtime timing dose noise temperature diet protein adults trial cohort effect").split()


def _corpus(seed: int, articles: int = 12) -> list[dict]:
    rnd = random.Random(seed)
    titles = ["Abstract", "Introduction", "Methods", "Results", "Discussion", "Conclusions", "References"]
    return [
        {"article_id": f"a{a}", "title": " ".join(rnd.sample(WORDS, 4)),
         "sections": [
             {"section_id": f"s{s}", "title": rnd.choice(titles),
              "text": " ".join(rnd.choices(WORDS, k=rnd.randint(5, 60))) + "."}
             for s in range(rnd.randint(1, 6))
         ]}
        for a in range(articles)
    ]


def _scan(index: Index, query: str) -> list[tuple[str, str, float]]:
    """BM25 прямым перебором всех разделов — эталон, с которым сверяется обратный индекс."""
    counts = []
    for section in index.sections:
        counts.append(Counter(
            tokenize(section.text) + tokenize(section.title) * retrieval.TITLE_WEIGHT
            + tokenize(section.article_title) * retrieval.ARTICLE_TITLE_WEIGHT
        ))
    n = len(counts)
    lengths = [sum(tf.values()) for tf in counts]
    average = sum(lengths) / n
    found = Counter(term for tf in counts for term in tf)
    scored = []
    for number, tf in enumerate(counts):
        score = 0.0
        norm = retrieval.K1 * (1 - retrieval.B + retrieval.B * lengths[number] / average)
        for term in Counter(tokenize(query)):
            if tf.get(term):
                idf = math.log(1 + (n - found[term] + 0.5) / (found[term] + 0.5))
                score += idf * tf[term] * (retrieval.K1 + 1) / (tf[term] + norm)
        if score > 0:
            section = index.sections[number]
            scored.append((section.article_id, section.section_id, score * retrieval.section_weight(section.title)))
    return sorted(scored, key=lambda item: -item[2])


def test_inverted_index_scores_sections_exactly_like_a_full_scan():
    rnd = random.Random(1)
    for seed in range(5):
        index = Index(_corpus(seed))
        for _ in range(40):
            query = " ".join(rnd.choices(WORDS, k=rnd.randint(1, 7))) + rnd.choice(["", " unknownword"])
            expected = _scan(index, query)
            # k больше числа статей и deep=0: поиск отдаёт лучший раздел каждой статьи в порядке статей
            hits = index.search(query, k=100, deep=0)
            best = {}
            for article_id, section_id, score in expected:
                best.setdefault(article_id, (section_id, score))
            assert {h.section.article_id: (h.section.section_id, h.score) for h in hits} == best  # оценки — до бита


def test_search_takes_second_section_only_from_the_best_articles():
    index = Index(_corpus(3))
    hits = index.search("caffeine sleep evening", k=6, deep=2)
    articles = [h.section.article_id for h in hits]
    assert len(hits) <= 6 and max(Counter(articles).values()) <= 2
    assert articles == sorted(articles, key=articles.index)  # разделы одной статьи стоят рядом
    assert index.search("", k=6) == [] and index.search("the of and", k=6) == []


def test_article_chosen_by_title_joins_the_ranking_even_without_matching_words():
    index = Index([
        {"article_id": "a1", "title": "Caffeine and sleep",
         "sections": [{"section_id": "s1", "title": "Results", "text": "Caffeine reduced sleep time."}]},
        {"article_id": "a2", "title": "Evening light",
         "sections": [{"section_id": "s1", "title": "Methods", "text": "Lamps were installed."},
                      {"section_id": "s2", "title": "Abstract", "text": "Light delayed melatonin."}]},
    ])
    hits = index.search("caffeine", k=6, prefer=["a2", "нет такой"])
    assert [(h.section.article_id, h.section.section_id) for h in hits] == [("a2", "s2"), ("a1", "s1")]


def test_abstract_is_found_by_title_and_replaced_by_first_section_when_missing():
    index = Index([
        {"article_id": "a1", "title": "T", "sections": [
            {"section_id": "s1", "title": "Introduction", "text": "x"},
            {"section_id": "s2", "title": "Abstract › Results", "text": "y"},
            {"section_id": "s3", "title": "Abstract › Conclusions", "text": "z"}]},
        {"article_id": "a2", "title": "T", "sections": [{"section_id": "s7", "title": "Methods", "text": "x"}]},
        {"article_id": "a3", "title": "T", "sections": []},
    ])
    assert index.abstract_of("a1").section_id == "s2"  # первая из нескольких
    assert index.abstract_of("a2").section_id == "s7"
    assert index.abstract_of("a3") is None and index.abstract_of("нет такой") is None


def test_excerpt_keeps_the_sentences_closest_to_the_question_in_their_order():
    filler = "The cohort was recruited from several clinics over a long period of observation. "
    text = filler * 4 + "Caffeine shortened sleep by 41 minutes. " + filler * 4 + "Caffeine also delayed sleep onset."
    index = Index([{"article_id": "a1", "title": "T", "sections": [{"section_id": "s1", "title": "Results", "text": text}]}])
    excerpt = index.focus(text, "caffeine sleep", max_chars=120)
    assert excerpt.startswith("Caffeine shortened sleep by 41 minutes.") and "delayed sleep onset" in excerpt
    assert len(excerpt) <= 120 + 200 and index.focus("Short text.", "caffeine", max_chars=120) == "Short text."


def test_sentences_with_the_same_matched_words_keep_their_order_in_the_text():
    """Равные по совпавшим словам предложения идут в порядке текста, как бы слова в них ни стояли.

    Сумма дробных весов зависит от порядка слагаемых, а порядок обхода множества у каждого процесса свой.
    Пока слова складывались в порядке множества, одна и та же вырезка менялась примерно в одном запуске из
    восьми (3 из 25 на отложенном наборе), а с ней и вход нейросети. Теперь слова складываются по алфавиту.
    """
    rnd = random.Random(11)
    terms = "alpha beta gamma delta kappa sigma omega lambda theta zeta rho tau".split()
    index = Index([])
    index._idf = {term: rnd.uniform(0.1, 3.0) for term in terms}
    for _ in range(200):
        first = rnd.sample(terms, 6)
        second = rnd.sample(first, 6)  # те же слова в другом порядке
        text = ("Nothing relevant here. " + " ".join(first).capitalize() + " one. "
                + " ".join(second).capitalize() + " two. " + "Nothing relevant here. " * 3)
        assert index.focus(text, " ".join(terms), max_chars=45).endswith("one.")
