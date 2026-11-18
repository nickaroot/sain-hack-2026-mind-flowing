"""Замена нейросети организаторов для локальной оценки: ответы берутся из файла.

Нужна, чтобы прогонять конвейер на больших наборах и не тратить суточный лимит
прокси. Вызов, которого нет в кеше, записывается в очередь
data/replay/<имя>.pending.jsonl. Ответы на очередь готовит другая модель
(при разработке — субагент Claude) и дописывает в data/replay/<имя>.jsonl,
после чего прогон повторяется. В решении на стенде этот класс не используется.
"""

import asyncio
import hashlib
import json
import re
from pathlib import Path

from app.llm import LLMError, Usage

ROOT = Path("data/replay")


def call_key(system: str, user: str) -> str:
    return hashlib.sha256(f"{system}\n\n{user}".encode()).hexdigest()[:16]


class ReplayLLM:
    deadline: float | None = None
    exhausted = False  # у локальной замены суточного лимита нет

    def __init__(self, name: str):
        ROOT.mkdir(parents=True, exist_ok=True)
        self.usage = Usage()
        self.pending = 0
        self._cache_path = ROOT / f"{name}.jsonl"
        self._pending_path = ROOT / f"{name}.pending.jsonl"
        self._cache: dict[str, dict] = {}
        if self._cache_path.exists():
            for line in self._cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._cache[row["key"]] = row["response"]
        self._pending_path.write_text("", encoding="utf-8")

    async def chat_json(self, system: str, user: str, *, max_tokens: int, effort: str = "low") -> dict:
        key = call_key(system, user)
        if key in self._cache:
            self.usage.calls += 1
            return self._cache[key]
        self.pending += 1
        with self._pending_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "system": system, "user": user}, ensure_ascii=False) + "\n")
        raise LLMError("ответа нет в локальном кеше")


class CodexLLM(ReplayLLM):
    """Та же модель, что у организаторов, но через Codex CLI и подписку ChatGPT разработчика.

    Используется только для локальной оценки: ни ключей, ни этого кода в работающем решении нет.
    Наша системная инструкция подставляется вместо встроенной инструкции агента Codex
    (настройка model_instructions_file), а данные запроса уходят пользовательским сообщением —
    как при вызове через прокси организаторов. Отличия, которые убрать нельзя: Codex прикладывает
    к запросу описания своих инструментов, не даёт задать температуру и не знает уровня minimal.
    Ответы кешируются в data/replay, повторный прогон с теми же запросами ничего не тратит.
    """

    BINARY = "/Applications/ChatGPT.app/Contents/Resources/codex-cli/bin/codex"
    WORKDIR = ROOT / "codex-empty"  # пустая папка: модель запускается в песочнице только для чтения
    TIMEOUT_SECONDS = 300
    SYSTEM_DIR = ROOT / "codex-system"
    ATTEMPTS = 3

    def __init__(self, name: str, model: str = "gpt-6-luna", parallel: int = 3):
        super().__init__(name)
        self.WORKDIR.mkdir(parents=True, exist_ok=True)
        self._model = model
        self._gate = asyncio.Semaphore(parallel)

    async def chat_json(self, system: str, user: str, *, max_tokens: int, effort: str = "low") -> dict:
        # Codex CLI изредка возвращает пустой ответ. У прокси организаторов повторы есть в app/llm.py,
        # здесь нужен свой, иначе один сбой искажает всю оценку.
        for attempt in range(self.ATTEMPTS):
            try:
                return await self._call(system, user, effort)
            except LLMError:
                if attempt == self.ATTEMPTS - 1:
                    raise
        raise LLMError("недостижимо")

    async def _call(self, system: str, user: str, effort: str) -> dict:
        key = call_key("codex-system-v2\n" + system, user)  # прежний кеш снят в другом режиме и сюда не подходит
        if key in self._cache:
            self.usage.calls += 1
            return self._cache[key]
        prompt = user
        system_path = self.SYSTEM_DIR / f"{call_key(system, '')}.txt"
        if not system_path.exists():
            self.SYSTEM_DIR.mkdir(parents=True, exist_ok=True)
            # Прокси получает response_format=json_object; здесь тот же запрет на лишний текст дан словами.
            system_path.write_text(
                system + "\n\nОтветь одним JSON-объектом без пояснений. Инструменты не вызывай.", encoding="utf-8"
            )
        out_path = self.WORKDIR / f"{key}.txt"
        async with self._gate:
            process = await asyncio.create_subprocess_exec(
                self.BINARY, "exec", "-m", self._model, "-s", "read-only", "--skip-git-repo-check",
                "--ephemeral", "-C", str(self.WORKDIR),
                # Напрямую модель не знает уровня minimal (прокси организаторов его принимает): ближайший — none.
                "-c", 'model_reasoning_effort="%s"' % ('none' if effort == 'minimal' else effort),
                "-c", 'model_instructions_file="%s"' % system_path.resolve(),
                "-o", str(out_path), "-",
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            )
            try:
                stdout, _ = await asyncio.wait_for(process.communicate(prompt.encode()), self.TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                process.kill()
                self.usage.failures += 1
                raise LLMError("Codex не ответил вовремя")
        used = re.search(r"tokens used\s*\n\s*([\d,]+)", stdout.decode(errors="replace"))
        self.usage.calls += 1
        self.usage.prompt_tokens += int(used.group(1).replace(",", "")) if used else 0
        try:
            text = out_path.read_text(encoding="utf-8").strip()
            out_path.unlink()
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise ValueError
        except (OSError, ValueError):
            (ROOT / "last-error.txt").write_text(stdout.decode(errors="replace")[-4000:], encoding="utf-8")
            self.usage.failures += 1
            raise LLMError("ответ Codex не разбирается как JSON")
        self._cache[key] = parsed
        with self._cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "response": parsed}, ensure_ascii=False) + "\n")
        return parsed
