"""Общие для тестов набор, готовые ответы и заглушка нейросети."""

import json

from app.llm import LLMError, Usage

EXPERTS = ["clinician", "biohacker", "nutritionist"]
# Текст обычной длины: проверка кодом возвращает слишком короткие и слишком длинные тексты на доработку.
BODY = "Кофе вечером мешает уснуть, в исследовании это видно по времени засыпания. " * 5
REFUSAL = "Количество и время приёма я не назову, с этим к врачу. " + BODY

DATASET = {
    "meta": {"dataset_id": "case4-test"},
    "chat_history": [],
    "experts": [{"expert_id": e, "role": "роль", "speech_style": "стиль"} for e in EXPERTS],
    "articles": [
        {"article_id": "a1", "title": "Evening light and melatonin",
         "sections": [{"section_id": "s1", "title": "Results", "text": "Evening light exposure suppressed melatonin."}]},
        {"article_id": "a2", "title": "Caffeine timing and sleep",
         "sections": [{"section_id": "s1", "title": "Results", "text": "Caffeine before bedtime reduced sleep time."}]},
    ],
    "requests": [
        {"record_id": f"r{i}", "user_id": "u1", "expert_id": expert, "group_id": "g1",
         "utterance": {"role": "user", "at": "2026-11-12T08:00:00+03:00", "text": "Мешает ли кофе вечером уснуть?"}}
        for i, expert in enumerate(EXPERTS)
    ],
}


class FakeLLM:
    """Отвечает как нейросеть: сначала поисковые слова, затем ответ на тройку."""

    deadline = None
    exhausted = False

    def __init__(self, *answers: dict, fail: bool = False, plan=None):
        self.usage = Usage()
        self.answers = list(answers)  # по порядку попыток; последний ответ повторяется
        self.fail = fail
        self.plan = plan or {"terms": "caffeine bedtime sleep time", "articles": []}
        self.payloads = []
        self.requests = []

    async def chat_json(self, system, user, **_):
        if self.fail:
            raise LLMError("недоступна")
        self.requests.append((system, user))  # дословно: по ним видно, что запрос к нейросети не изменился
        data = json.loads(user)
        if "catalog" in data:
            self.catalog = data["catalog"]
            return {item["id"]: self.plan for item in data["items"]}
        self.payloads.append(data)
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


def good_answer(**overrides) -> dict:
    base = {"asks": [{"q": "мешает ли кофе вечером", "ok": True}], "forbidden": "none", "kind": "answer",
            "source": 1, "quote": "Caffeine before bedtime", "texts": {e: f"{BODY} Подпись {e}." for e in EXPERTS}}
    return {**base, **overrides}
