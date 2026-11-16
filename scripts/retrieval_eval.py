"""Оценка поиска на наборе с эталоном без траты суточного лимита.

План поиска (поисковые слова и статьи, выбранные по названиям) берётся из локального кеша Codex CLI:
первый запуск делает один-три вызова, дальше поиск можно настраивать бесплатно.

Запуск: python -m scripts.retrieval_eval data/case4-dev-v2.json dev-plan
"""

import asyncio
import json
import sys
from collections import defaultdict
from datetime import datetime

from app import pipeline
from app.retrieval import Index
from scripts.offline_llm import CodexLLM


async def main() -> None:
    with open(sys.argv[1], encoding="utf-8") as f:
        bundle = json.load(f)
    dataset = bundle["dataset"]
    allowed = {r["record_id"]: set(r["allowed_article_ids"]) for r in bundle["reference"]["requests"]}
    by_user = defaultdict(list)
    for message in sorted(dataset["chat_history"], key=lambda m: datetime.fromisoformat(m["at"])):
        by_user[message["user_id"]].append(message)
    groups = pipeline._group_requests(dataset["requests"])
    histories = [pipeline._history_for(g[0], by_user) for g in groups]
    plans = await pipeline._search_plans(CodexLLM(sys.argv[2]), groups, histories, dataset["articles"])

    index = Index(dataset["articles"])
    totals = defaultdict(int)
    for group, plan in zip(groups, plans):
        ok = allowed[group[0]["record_id"]]
        for name, prefer in (("только слова", ()), ("слова и названия", plan.articles)):
            articles = [h.section.article_id for h in index.search(plan.terms, k=pipeline.CANDIDATES, prefer=prefer)]
            totals[name, "первая"] += bool(articles) and articles[0] in ok
            totals[name, "среди кандидатов"] += any(a in ok for a in articles)
        mark = "+" if articles and articles[0] in ok else ("~" if any(a in ok for a in articles) else "-")
        print(mark, sorted(ok), "<-", list(dict.fromkeys(articles)), "| по названиям:", plan.articles)
    for name in ("только слова", "слова и названия"):
        print(f"{name}: допустимая статья первая {totals[name, 'первая']}/{len(groups)}, "
              f"среди кандидатов {totals[name, 'среди кандидатов']}/{len(groups)}")


if __name__ == "__main__":
    asyncio.run(main())
