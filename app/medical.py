"""Словарь проверки медицинских формулировок, опубликованный организаторами.

Их проверка сначала применяет этот словарь и только потом модель; совпадение со словарём означает,
что ответ не будет засчитан, а для категории judge_manipulation — ещё и разбор организаторами.
Файл docs/api/medcheck-resources.json взят без изменений с сайта документации хакатона
(https://sainorg.github.io/sain-hackathon-docs/medcheck-resources.json).
"""

import json
import re
from pathlib import Path

_RESOURCES = Path(__file__).parent.parent / "docs" / "api" / "medcheck-resources.json"


def _compile(expression: str) -> re.Pattern:
    # Словарь написан для движка с классами \p{L} и \p{N}; в Python «буква, цифра или _» — это \w.
    converted = expression.replace(r"(?<![\p{L}\p{N}_])", r"(?<!\w)").replace(r"(?![\p{L}\p{N}_])", r"(?!\w)")
    return re.compile(converted, re.IGNORECASE)


def _load() -> list[tuple[str, re.Pattern]]:
    try:
        entries = json.loads(_RESOURCES.read_text(encoding="utf-8"))["dictionary"]
        return [(entry["category"], _compile(entry["expression"])) for entry in entries]
    except (OSError, ValueError, KeyError, re.error):
        return []  # без словаря остаются собственные фильтры решения


DICTIONARY = _load()


def dictionary_match(text: str) -> tuple[str, str] | None:
    """Категория и точная цитата, если текст попадает под словарь организаторов."""
    for category, pattern in DICTIONARY:
        match = pattern.search(text)
        if match:
            return category, match.group(0).strip()
    return None
