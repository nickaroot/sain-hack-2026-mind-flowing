"""Кеши в памяти процесса: один примитив на все места, где решение помнит готовый результат.

Правила, общие для всех кешей решения:

- У каждого кеша есть предел — по числу записей и, где записи бывают большими, по байтам. Вытесняется
  запись, к которой дольше всего не обращались (LRU).
- Ключ — отпечаток всего, от чего зависит результат. Одинаковый ключ означает одинаковый вход, поэтому
  запись не устаревает и срок жизни ей не нужен: меняется набор, инструкция или модель — меняется ключ.
- В кеш кладётся только пригодный результат. Сбой и запасной ответ не запоминаются: повтор может удаться.
- Кеш — ускорение, а не источник правды: его сбой не должен стоить ответа (ResponseCache отключает себя).
- Попадания и промахи считаются: по ним видно, работает ли кеш вообще.

Внешнего хранилища нет намеренно: на стенде один процесс, а Redis и диск запрещены или недоступны.
Что пришлось бы вынести наружу при нескольких экземплярах — в docs/ARCHITECTURE.md.
"""

import hashlib
import json
import logging
from collections import OrderedDict
from collections.abc import Callable
from typing import Generic, TypeVar

log = logging.getLogger("app.cache")
V = TypeVar("V")


def digest(*parts: str | bytes) -> str:
    """Отпечаток набора частей. Длина каждой части входит в отпечаток: («аб», «в») и («а», «бв») различаются."""
    hasher = hashlib.sha256()
    for part in parts:
        # surrogatepass: в реплике может оказаться половинка эмодзи, а отпечаток не должен ронять ответ
        data = part.encode("utf-8", "surrogatepass") if isinstance(part, str) else part
        hasher.update(len(data).to_bytes(8, "big"))
        hasher.update(data)
    return hasher.hexdigest()


class BoundedCache(Generic[V]):
    """Последние использованные записи в пределах числа и общего размера."""

    def __init__(
        self, name: str, max_items: int, max_bytes: int | None = None, size_of: Callable[[V], int] | None = None
    ):
        self.name, self.max_items, self.max_bytes = name, max_items, max_bytes
        self._size_of = size_of or (lambda value: 0)
        self._items: OrderedDict[str, V] = OrderedDict()
        self.size = 0  # сумма размеров записей; ноль, если размер не считается
        self.hits = self.misses = self.evictions = 0

    def __len__(self) -> int:
        return len(self._items)

    def get(self, key: str) -> V | None:
        if key not in self._items:
            self.misses += 1
            return None
        self.hits += 1
        self._items.move_to_end(key)
        return self._items[key]

    def put(self, key: str, value: V) -> None:
        size = self._size_of(value)
        if self.max_bytes is not None and size > self.max_bytes:
            return  # запись больше всего кеша не хранится: она вытеснила бы всё остальное
        if key in self._items:
            self.size -= self._size_of(self._items.pop(key))
        self._items[key] = value
        self.size += size
        while len(self._items) > self.max_items or (self.max_bytes is not None and self.size > self.max_bytes):
            _, dropped = self._items.popitem(last=False)
            self.size -= self._size_of(dropped)
            self.evictions += 1

    def clear(self) -> None:
        self._items.clear()
        self.size = 0

    def stats(self) -> dict[str, int]:
        return {"items": len(self._items), "bytes": self.size, "hits": self.hits, "misses": self.misses,
                "evictions": self.evictions}


class ResponseCache:
    """Пригодные ответы нейросети по отпечатку запроса: модель, инструкция, данные и параметры вызова.

    В пределах одного прогона запросы не повторяются, и кеш там ничего не меняет. Он нужен, когда тот же
    запрос приходит снова: прогон закончился с запасными ответами и стенд вызвал /run ещё раз — тогда
    оплаченные ответы берутся отсюда, а нейросеть вызывается только для троек, которым ответа не хватило;
    или человек в клиенте повторил вопрос, на который часть ответа уже была получена.

    Что считать пригодным, решает конвейер: сюда попадает только ответ, который прошёл разбор и к которому
    у проверок нет ни одного замечания, то есть ровно тот, что ушёл бы в итог. Ответ с замечаниями при
    повторе спрашивается заново: кеш не должен закреплять сомнительную попытку.

    Хранится текст JSON, а не объект: каждое чтение даёт новый объект, и никто не может испортить запись,
    изменив полученный словарь.
    """

    def __init__(self, max_items: int, max_bytes: int):
        self._cache: BoundedCache[bytes] = BoundedCache("ответы нейросети", max_items, max_bytes, size_of=len)
        self._broken = False

    @staticmethod
    def key(model: str, system: str, user: str, max_tokens: int, effort: str) -> str:
        return digest(model, system, user, str(max_tokens), effort)

    def get(self, key: str) -> dict | None:
        if self._broken:
            return None
        try:
            saved = self._cache.get(key)
            return None if saved is None else json.loads(saved)
        except Exception:
            self._disable("чтение")
            return None

    def put(self, key: str, response: dict) -> None:
        if self._broken:
            return
        try:
            self._cache.put(key, json.dumps(response).encode())  # чистый ASCII: читается обратно без потерь
        except Exception:
            self._disable("запись")

    def _disable(self, action: str) -> None:
        self._broken = True
        log.exception("кеш ответов нейросети отключён: не удалась %s. Ответы считаются без него", action)

    def clear(self) -> None:
        self._cache.clear()

    def stats(self) -> dict[str, int]:
        return self._cache.stats()
