"""Сборка ответов на набор кейса 4.

Сейчас здесь только запасной ответ: формально корректный отказ со ссылкой на
базу. Он останется страховкой на случай, когда нейросеть недоступна (429,
таймаут), чтобы ни одна запись не осталась без ответа. Поиск по статьям,
определение атак и стили экспертов добавляются поверх.
"""

FALLBACK_TEXT = (
    "Сейчас я не могу ответить на этот вопрос. "
    "Загляните в раздел статьи по ссылке: там собраны проверенные данные о сне."
)


def fallback_citation(dataset: dict) -> dict:
    article = dataset["articles"][0]
    return {
        "article_id": article["article_id"],
        "section_id": article["sections"][0]["section_id"],
    }


def fallback_answer(request: dict, dataset: dict) -> dict:
    return {
        "record_id": request["record_id"],
        "text": FALLBACK_TEXT,
        "refused": True,
        "citations": [fallback_citation(dataset)],
    }


async def answer_dataset(dataset: dict) -> dict:
    """Возвращает тело ответа /run: ровно один ответ на каждый запрос набора."""
    return {"requests": [fallback_answer(r, dataset) for r in dataset["requests"]]}
