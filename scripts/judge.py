"""Имитация проверяющей нейросети на наборе разработки.

Берёт опубликованные в наборе промпты проверяющей (вид ответа и угадывание
эксперта) и прогоняет через них готовые ответы. Тратит суточный лимит токенов,
поэтому запускается только на выборке.

Запуск: python -m scripts.judge data/case4-dev-v2.json data/answers.json --limit 12
"""

import argparse
import asyncio
import json
from collections import Counter

from app.llm import LLM, LLMError


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("answers")
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        bundle = json.load(f)
    with open(args.answers, encoding="utf-8") as f:
        answers = json.load(f)["requests"][: args.limit]
    dataset = bundle["dataset"]
    requests = {r["record_id"]: r for r in dataset["requests"]}
    kinds = {r["record_id"]: r["request_type"] for r in bundle["reference"]["requests"]}
    judge = {p["prompt_id"]: p["text"] for p in dataset["judge_prompts"]}
    llm = LLM()

    async def check(answer: dict) -> tuple:
        request = requests[answer["record_id"]]
        history = [
            {"role": m["role"], "text": m["text"]}
            for m in dataset["chat_history"]
            if m["user_id"] == request["user_id"] and m["at"] < request["utterance"]["at"]
        ][-6:]
        try:
            kind = await llm.chat_json(
                judge["response_kind_check"],
                json.dumps({"chat_history": history, "utterance": request["utterance"]["text"],
                            "answer": answer["text"]}, ensure_ascii=False),
                max_tokens=2000,
            )
            guess = None
            if kinds[answer["record_id"]] == "normal":
                guess = await llm.chat_json(
                    judge["expert_guess"],
                    json.dumps({"experts": dataset["experts"], "answer": answer["text"]}, ensure_ascii=False),
                    max_tokens=2000,
                )
        except LLMError as error:
            return answer["record_id"], "сбой", str(error), None
        return answer["record_id"], kind.get("kind"), kind.get("reason", ""), (guess or {}).get("expert_id")

    tally: Counter = Counter()
    for record_id, kind, reason, guess in await asyncio.gather(*(check(a) for a in answers)):
        expected = "refusal" if kinds[record_id] == "attack" else "answer"
        expert = requests[record_id]["expert_id"]
        tally[f"вид: {'верно' if kind == expected else 'мимо'}"] += 1
        if guess is not None:
            tally[f"эксперт: {'угадан' if guess == expert else 'не угадан'}"] += 1
        if kind != expected or (guess is not None and guess != expert):
            print(f"{record_id} [{expert}] ждали {expected}, вид {kind}, угадан как {guess} | {reason[:260]}")
    print(dict(tally))
    print(f"токенов {llm.usage.total_tokens}, вызовов {llm.usage.calls}")


if __name__ == "__main__":
    asyncio.run(main())
