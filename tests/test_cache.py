"""Общий примитив кешей: пределы, вытеснение, счётчики и память пригодных ответов нейросети."""

import asyncio

from app import pipeline
from app.cache import BoundedCache, ResponseCache, digest
from tests.fakes import BODY, DATASET, EXPERTS, FakeLLM, good_answer


def test_cache_is_bounded_by_bytes_as_well_as_by_count():
    cache: BoundedCache[bytes] = BoundedCache("тест", max_items=3, max_bytes=10, size_of=len)
    cache.put("a", b"1234")
    cache.put("b", b"5678")
    cache.put("c", b"90ab")  # 12 байт на три записи: вытесняется та, к которой дольше всех не обращались
    assert cache.get("a") is None and cache.get("b") == b"5678" and cache.size == 8 and len(cache) == 2
    cache.put("d", b"x" * 11)  # запись больше всего кеша не хранится и ничего не вытесняет
    assert cache.get("d") is None and len(cache) == 2
    for key in "efg":
        cache.put(key, b"1")
    assert len(cache) == 3 and cache.size == 3
    cache.put("g", b"12345")  # замена записи пересчитывает размер
    assert cache.size == 7 and cache.get("g") == b"12345"


def test_least_recently_used_entry_goes_first_and_hits_are_counted():
    cache: BoundedCache[int] = BoundedCache("тест", max_items=2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1  # «a» только что читали — вытеснится «b»
    cache.put("c", 3)
    assert cache.get("b") is None and cache.get("a") == 1 and cache.get("c") == 3
    assert cache.stats() == {"items": 2, "bytes": 0, "hits": 3, "misses": 1, "evictions": 1}


def test_digest_does_not_glue_parts_together():
    assert digest("аб", "в") != digest("а", "бв") and digest("a", b"b") == digest("a", "b")
    key = ResponseCache.key("m", "инструкция", "данные", 3500, "low")
    assert key != ResponseCache.key("m2", "инструкция", "данные", 3500, "low")  # другая модель — другой ответ
    assert key != ResponseCache.key("m", "инструкция", "данные", 3500, "minimal")


def test_response_cache_returns_a_fresh_object_and_fails_open():
    cache = ResponseCache(max_items=10, max_bytes=10_000)
    cache.put("k", {"texts": {"a": "б"}})
    first = cache.get("k")
    first["texts"]["a"] = "испорчено"  # вызывающий код не может испортить запись
    assert cache.get("k") == {"texts": {"a": "б"}}
    cache.put("bad", {"x": {1, 2}})  # не сериализуется: кеш отключает себя, а не роняет ответ
    assert cache.get("k") is None and cache.stats()["items"] == 1
    cache.put("k2", {"ok": True})
    assert cache.get("k2") is None


def _answers(llm, cache):
    return asyncio.run(pipeline.answer_dataset(DATASET, llm, cache=cache))


def test_repeated_dataset_takes_usable_answers_from_the_cache():
    cache = ResponseCache(max_items=100, max_bytes=1_000_000)
    first = _answers(FakeLLM(good_answer()), cache)
    again = FakeLLM(good_answer())
    assert _answers(again, cache) == first and again.requests == []  # ни плана, ни ответа второй раз не спрашивали
    assert cache.stats()["hits"] == 2 and cache.stats()["items"] == 2


def test_answer_with_remarks_is_asked_again_and_only_the_clean_one_is_remembered():
    long = good_answer(texts={e: BODY * 4 for e in EXPERTS})  # слишком длинный текст — замечание к подаче
    cache = ResponseCache(max_items=100, max_bytes=1_000_000)
    first_llm = FakeLLM(long, good_answer())
    first = _answers(first_llm, cache)
    assert len(first_llm.payloads) == 2 and "note" in first_llm.payloads[1]
    assert cache.stats()["items"] == 2  # план и исправленный ответ; попытка с замечаниями не запомнена
    again = FakeLLM(long, good_answer())
    assert _answers(again, cache) == first
    assert len(again.payloads) == 1 and "note" not in again.payloads[0]  # первую попытку спросили заново


def test_unusable_answers_are_asked_again_instead_of_being_remembered():
    cache = ResponseCache(max_items=100, max_bytes=1_000_000)
    dose = good_answer(texts={e: f"{BODY} Хватит 100 мг кофеина." for e in EXPERTS})  # небезопасно — запасной отказ
    fallback = _answers(FakeLLM(dose), cache)
    assert all(pipeline.is_fallback(a) for a in fallback["requests"])
    malformed = FakeLLM({"texts": "не словарь", "source": 1})
    _answers(malformed, cache)
    assert len(malformed.payloads) == 2  # ответ не по форме тоже не запомнен: вторая попытка ушла нейросети
    assert cache.stats()["items"] == 1  # в памяти только план поиска
    repaired = FakeLLM(good_answer())
    assert not any(pipeline.is_fallback(a) for a in _answers(repaired, cache)["requests"])
    assert len(repaired.payloads) == 1 and len(repaired.requests) == 1  # план взят из памяти, ответ спрошен заново


def test_only_the_missing_part_of_a_degraded_run_is_paid_for_again():
    """Прогон, где нейросеть отказала посреди набора, повторяется: готовые тройки второй раз не оплачиваются."""
    requests = []
    for group, text in (("g1", "Мешает ли кофе вечером уснуть?"), ("g2", "А чай перед сном?"), ("g3", "А какао?")):
        requests += [{"record_id": f"{group}-{e}", "user_id": "u1", "expert_id": e, "group_id": group,
                      "utterance": {"role": "user", "at": "2026-11-12T08:00:00+03:00", "text": text}} for e in EXPERTS]
    dataset = {**DATASET, "requests": requests}

    class Outage(FakeLLM):
        """Отвечает на план и на первую тройку, дальше недоступна."""

        async def chat_json(self, system, user, **options):
            if "catalog" not in user and len(self.payloads) >= 1:
                from app.llm import LLMError
                raise LLMError("прокси недоступен")
            return await super().chat_json(system, user, **options)

    cache = ResponseCache(max_items=100, max_bytes=1_000_000)
    degraded = asyncio.run(pipeline.answer_dataset(dataset, Outage(good_answer()), cache=cache))
    assert sum(pipeline.is_fallback(a) for a in degraded["requests"]) == 6
    again = FakeLLM(good_answer())
    repeated = asyncio.run(pipeline.answer_dataset(dataset, again, cache=cache))
    assert not any(pipeline.is_fallback(a) for a in repeated["requests"])
    assert len(again.requests) == 2  # оплачены только две тройки, которым ответа не хватило
    assert repeated["requests"][:3] == degraded["requests"][:3]


def test_surrogate_half_in_a_question_does_not_break_the_key():
    assert ResponseCache.key("m", "s", "половинка эмодзи \ud83d", 1, "low")
    cache = ResponseCache(max_items=10, max_bytes=10_000)
    cache.put("k", {"text": "\ud83d"})
    assert cache.get("k") == {"text": "\ud83d"}


def test_split_of_a_search_plan_does_not_send_an_empty_batch():
    requests = [{"record_id": f"r{i}", "user_id": "u1", "expert_id": "clinician", "group_id": f"g{i}",
                 "utterance": {"role": "user", "at": "2026-11-12T08:00:00+03:00", "text": f"Вопрос {i}?"}} for i in range(3)]

    class OneMissing(FakeLLM):
        async def chat_json(self, system, user, **options):
            import json as _json
            data = _json.loads(user)
            if "catalog" not in data:
                return await super().chat_json(system, user, **options)
            self.requests.append((system, user))
            assert data["items"], "пустая пачка ушла нейросети"
            ids = [item["id"] for item in data["items"]]
            return {i: self.plan for i in (ids[:-1] if len(ids) == 3 else ids)}

    llm = OneMissing(good_answer())
    asyncio.run(pipeline.answer_dataset({**DATASET, "requests": requests}, llm))
    assert len([1 for _, user in llm.requests if "catalog" in user]) == 2  # вся пачка и одна недостающая реплика


def test_incomplete_search_plan_is_not_remembered():
    cache = ResponseCache(max_items=100, max_bytes=1_000_000)

    class NoPlan(FakeLLM):
        async def chat_json(self, system, user, **options):
            self.requests.append((system, user))
            return {} if "catalog" in user else good_answer()

    _answers(NoPlan(good_answer()), cache)
    assert cache.stats()["items"] == 0
    recovered = FakeLLM(good_answer())
    assert not any(pipeline.is_fallback(a) for a in _answers(recovered, cache)["requests"])
    assert len(recovered.requests) == 2


def test_without_a_cache_every_request_goes_to_the_model():
    llm = FakeLLM(good_answer())
    asyncio.run(pipeline.answer_dataset(DATASET, llm))
    asyncio.run(pipeline.answer_dataset(DATASET, llm))
    assert len(llm.requests) == 4
