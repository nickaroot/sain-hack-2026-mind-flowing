"""Логи: общий формат и метка прогона в каждой строке.

Прогон набора — это десятки параллельных вызовов нейросети. Метка прогона ставится один раз в задаче
прогона и через contextvars сама попадает во все строки, которые пишут её подзадачи: по ней из логов
стенда выбирается всё, что относится к одному /run. В логи не пишутся токены доступа, код набора,
реплики пользователей и тексты запросов к нейросети.
"""

import logging
import uuid
from contextvars import ContextVar

_run: ContextVar[str] = ContextVar("run", default="-")


class _RunMark(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run = _run.get()
        return True


def setup() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(run)s] %(message)s")
    for handler in logging.getLogger().handlers:
        if not any(isinstance(f, _RunMark) for f in handler.filters):
            handler.addFilter(_RunMark())
    # httpx на уровне INFO пишет URL запроса, а в нём код набора: в логи он попадать не должен.
    logging.getLogger("httpx").setLevel(logging.WARNING)


def start_run() -> str:
    """Помечает текущую задачу и всё, что она запустит, новой меткой прогона."""
    mark = uuid.uuid4().hex[:8]
    _run.set(mark)
    return mark
