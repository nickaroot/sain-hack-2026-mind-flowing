"""Разбор промахов: для каждой тройки с ошибкой показывает реплику, поисковые слова, кандидатов и выбор.

Запуск: python -m scripts.diagnose data/heldout.json data/heldout-answers.json heldout-gen
Работает только по локальному кешу Codex и ничего не тратит.
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
    with open(sys.argv[2], encoding="utf-8") as f:
        answers = {a["record_id"]: a for a in json.load(f)["requests"]}
    dataset = bundle["dataset"]
    ref = {r["record_id"]: r for r in bundle["reference"]["requests"]}
    by_user = defaultdict(list)
    for m in sorted(dataset["chat_history"], key=lambda m: datetime.fromisoformat(m["at"])):
        by_user[m["user_id"]].append(m)
    groups = pipeline._group_requests(dataset["requests"])
    histories = [pipeline._history_for(g[0], by_user) for g in groups]
    plans = await pipeline._search_plans(CodexLLM(sys.argv[3]), groups, histories, dataset["articles"])
    index = Index(dataset["articles"])
    for group, plan in zip(groups, plans):
        query = plan.terms
        first = group[0]["record_id"]
        expected, answer = ref[first], answers[first]
        attack = expected["request_type"] == "attack"
        cited = answer["citations"][0]["article_id"]
        flag_ok, cite_ok = answer["refused"] == attack, cited in expected["allowed_article_ids"]
        if flag_ok and cite_ok:
            continue
        hits = [h.section.article_id for h in index.search(query, k=pipeline.CANDIDATES, prefer=plan.articles)]
        print(f"\n[{'флаг' if not flag_ok else 'ссылка'}] {expected['rationale'][:70]}")
        print("  реплика:", group[0]["utterance"]["text"][:260])
        print("  слова:", query[:200])
        print(f"  выбор по названиям: {plan.articles}")
        print(f"  кандидаты: {hits}  выбрано: {cited}  допустимо: {expected['allowed_article_ids']}")
        print("  ответ:", answer["text"][:200])


if __name__ == "__main__":
    asyncio.run(main())
