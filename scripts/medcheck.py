"""Локальная копия проверки медицинских формулировок организаторов.

Промпт, словарь и примеры опубликованы организаторами (docs/api/medcheck-resources.json, источник —
https://sainorg.github.io/sain-hackathon-docs/medcheck-resources.json). Порядок такой же, как в их
ответе: сначала словарь (source=dictionary), затем модель (source=model). Модель та же, gpt-6-luna,
но вызывается через Codex CLI, где температуру задать нельзя, поэтому вердикты совпадают не всегда:
насколько — показывает сверка с настоящими вердиктами (ключ --calibrate).

Запуск: python -m scripts.medcheck data/answers.json medcheck-local
        python -m scripts.medcheck --calibrate data/medcheck-real.json medcheck-local
"""

import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

from app import medical
from app.llm import LLMError
from scripts.offline_llm import CodexLLM

RESOURCES = json.loads((Path(__file__).parent.parent / "docs/api/medcheck-resources.json").read_text(encoding="utf-8"))
CATEGORIES = ("clean", "diagnosis", "treatment", "dosage", "real_person_name", "judge_manipulation")


def dictionary_verdict(text: str) -> dict | None:
    found = medical.dictionary_match(text)  # тот же словарь работает и в самом решении
    return {"category": found[0], "quote": found[1], "source": "dictionary"} if found else None


async def verdict(llm: CodexLLM, text: str) -> dict:
    found = dictionary_verdict(text)
    if found:
        return found
    try:
        answer = await llm.chat_json(RESOURCES["prompt"]["text"], text, max_tokens=1500, effort="low")
    except LLMError as error:
        return {"category": None, "quote": None, "source": f"сбой: {error}"}
    category = answer.get("category")
    return {"category": category if category in CATEGORIES else None, "quote": answer.get("quote"), "source": "model"}


async def main() -> None:
    calibrate = sys.argv[1] == "--calibrate"
    path, cache = (sys.argv[2], sys.argv[3]) if calibrate else (sys.argv[1], sys.argv[2])
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    rows = data if calibrate else data["requests"]
    llm = CodexLLM(cache)
    verdicts = await asyncio.gather(*(verdict(llm, row["text"]) for row in rows))
    if calibrate:
        same = sum(v["category"] == row["category"] for v, row in zip(verdicts, rows, strict=True))
        flagged = sum((v["category"] != "clean") == (row["category"] != "clean") for v, row in zip(verdicts, rows, strict=True))
        print(f"совпало категорий {same}/{len(rows)}; совпало «чисто или нет» {flagged}/{len(rows)}")
        for v, row in zip(verdicts, rows, strict=True):
            mark = "=" if v["category"] == row["category"] else "≠"
            print(f"{mark} настоящий {row['category']:<10} локальный {v['category']!s:<10} | {str(v['quote'])[:110]}")
    else:
        print(dict(Counter(str(v["category"]) for v in verdicts)))
        for v, row in zip(verdicts, rows, strict=True):
            if v["category"] != "clean":
                print(f"- {row['record_id']}: {v['category']} ({v['source']}) | {str(v['quote'])[:160]}")
    print(f"вызовов {llm.usage.calls}, сбоев {llm.usage.failures}")


if __name__ == "__main__":
    asyncio.run(main())
