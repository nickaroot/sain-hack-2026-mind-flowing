"""Оценка поиска на наборе разработки без траты токенов.

Поисковые слова берутся из кеша data/queries-dev.json. Если кеша нет, делается
один вызов нейросети на весь набор (около 3 тыс. токенов), дальше поиск можно
настраивать бесплатно.

Запуск: python -m scripts.retrieval_eval data/case4-dev-v2.json
"""

import asyncio
import json
import os
import sys
from collections import defaultdict

from app import pipeline
from app.llm import LLM
from app.retrieval import Index

CACHE = "data/queries-dev.json"


async def main() -> None:
    with open(sys.argv[1], encoding="utf-8") as f:
        bundle = json.load(f)
    dataset = bundle["dataset"]
    allowed = {r["record_id"]: set(r["allowed_article_ids"]) for r in bundle["reference"]["requests"]}
    by_user = defaultdict(list)
    for message in sorted(dataset["chat_history"], key=lambda m: m["at"]):
        by_user[message["user_id"]].append(message)
    groups = pipeline._group_requests(dataset["requests"])
    histories = [pipeline._history_for(g[0], by_user) for g in groups]

    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            queries = json.load(f)
    else:
        llm = LLM()
        queries = await pipeline._search_queries(llm, groups, histories)
        with open(CACHE, "w", encoding="utf-8") as f:
            json.dump(queries, f, ensure_ascii=False, indent=1)
        print("поисковые слова получены, токенов:", llm.usage.total_tokens)

    index = Index(dataset["articles"])
    top1 = topk = 0
    for group, query in zip(groups, queries):
        ok = allowed[group[0]["record_id"]]
        hits = index.search(query, k=pipeline.CANDIDATES)
        articles = [h.section.article_id for h in hits]
        top1 += bool(articles) and articles[0] in ok
        topk += any(a in ok for a in articles)
        mark = "+" if articles and articles[0] in ok else ("~" if any(a in ok for a in articles) else "-")
        print(mark, sorted(ok), "<-", articles, "|", [h.section.title[:18] for h in hits[:3]])
    print(f"первая статья допустима: {top1}/{len(groups)}; допустимая есть среди кандидатов: {topk}/{len(groups)}")


if __name__ == "__main__":
    asyncio.run(main())
