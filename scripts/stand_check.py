"""Прогон решения в условиях стенда: контейнер с теми же ограничениями и эмулятор внешних сервисов.

Повторяет то, что делает стенд организаторов (по их инструкции и README архива команды):
1. собирает образ python:3.11-slim с пакетами из requirements.txt (сеть есть только при сборке);
2. запускает start.sh от пользователя 10001 с 512 МБ памяти, 1 CPU, 256 процессами, запись только в /tmp;
   имена сайтов не резолвятся — доступны только эмулятор сервиса хакатона и прокси;
3. каждые 2 секунды до 60 секунд ждёт любой HTTP-ответ на GET /;
4. один раз вызывает POST /run с токеном прогона и кодом набора и ждёт ответ до 45 минут;
5. проверяет то, за что автоматическая часть обнуляется: HTTP 200, JSON, размер, формат (validate).
Дополнительно: повторный /run не должен тратить лимит второй раз, а после прогона серверная часть должна
отвечать клиенту — как на финальном показе: список экспертов, разговор, запрос из браузера с другого адреса.

Нужен Docker. Запуск: python -m scripts.stand_check --dataset case4-dev-v2=data/case4-dev-v2.json --cache emu-dev
Любые ключи эмулятора (--spent, --reject, --fault-rate, --rpm) можно дописать в конец команды.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx

from scripts.emulator import LLM_KEY, MODEL, RUN_TOKEN, TEAM_TOKEN

ROOT = Path(__file__).parent.parent
IMAGE, CONTAINER = "mindflowing-stand", "mindflowing-stand-check"
APP_PORT = 18000
MAX_BODY_BYTES = 32 * 1024 * 1024  # сайт документации называет 32 МиБ, инструкция — 64 МБ: берём меньшее
DOCKERFILE = """FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -q --no-cache-dir -r requirements.txt
COPY . .
"""


def sh(*command: str, check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=check, text=True, capture_output=True, **kwargs)


def build_image() -> None:
    """Образ из тех же файлов, что уйдут на стенд: git-архив рабочей копии без игнорируемых файлов."""
    with tempfile.TemporaryDirectory() as context:
        files = sh("git", "ls-files", "-c", "-o", "--exclude-standard", cwd=ROOT).stdout.split("\n")
        archive = subprocess.run(["tar", "-cf", "-", *[f for f in files if f and (ROOT / f).exists()]],
                                 cwd=ROOT, check=True, capture_output=True).stdout
        subprocess.run(["tar", "-xf", "-", "-C", context], input=archive, check=True)
        (Path(context) / "Dockerfile").write_text(DOCKERFILE)
        result = sh("docker", "build", "-q", "-t", IMAGE, context, check=False)
        if result.returncode:
            sys.exit("сборка образа не удалась:\n" + result.stderr[-2000:])


def start_container(emulator_port: int) -> None:
    sh("docker", "rm", "-f", CONTAINER, check=False)
    base = f"http://host.docker.internal:{emulator_port}"
    sh(
        "docker", "run", "-d", "--name", CONTAINER,
        "--memory", "512m", "--memory-swap", "512m", "--cpus", "1", "--pids-limit", "256",
        "--user", "10001", "--read-only", "--tmpfs", "/tmp:size=256m",
        "--dns", "127.0.0.1",  # как на стенде: посторонние имена не резолвятся
        "--add-host", "host.docker.internal:host-gateway",
        "-p", f"{APP_PORT}:8000",
        "-e", "PORT=8000", "-e", "TEAM=team23", "-e", f"HACK_API_URL={base}", "-e", f"HACK_TEAM_TOKEN={TEAM_TOKEN}",
        "-e", f"LLM_BASE_URL={base}/v1", "-e", f"LLM_API_KEY={LLM_KEY}", "-e", f"LLM_MODEL={MODEL}",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        IMAGE, "sh", "start.sh",
    )


def wait_ready(seconds: int = 60) -> float:
    started = time.time()
    while time.time() - started < seconds:
        try:
            httpx.get(f"http://127.0.0.1:{APP_PORT}/", timeout=2)
            return time.time() - started
        except httpx.HTTPError:
            time.sleep(2)
    logs = sh("docker", "logs", "--tail", "50", CONTAINER, check=False)
    sys.exit("сервер не ответил на GET / за 60 секунд — на стенде деплой откатился бы:\n" + logs.stdout + logs.stderr)


def watch_memory(peak: dict, stop: threading.Event) -> None:
    while not stop.is_set():
        usage = sh("docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", CONTAINER, check=False).stdout
        value = usage.split("/")[0].strip()
        if value.endswith("MiB"):
            peak["mib"] = max(peak["mib"], float(value[:-3]))
        elif value.endswith("GiB"):
            peak["mib"] = max(peak["mib"], float(value[:-3]) * 1024)
        stop.wait(3)


def check_client_methods(emulator_port: int) -> list[str]:
    """То, что делает клиент на финальном показе: после /run спрашивает экспертов и задаёт вопрос из браузера."""
    base, problems = f"http://127.0.0.1:{APP_PORT}/api", []
    tokens = lambda: httpx.get(f"http://127.0.0.1:{emulator_port}/_emulator/stats", timeout=10).json()["tokens"]  # noqa: E731
    experts = httpx.get(f"{base}/experts", timeout=30)
    if experts.status_code != 200 or len(experts.json().get("experts", [])) < 3:
        return [f"/api/experts: HTTP {experts.status_code}"]
    ids = [e["expert_id"] for e in experts.json()["experts"]]
    question = {"expert_id": ids[0], "text": "Мешает ли чашка кофе после обеда уснуть вечером?"}
    before, started = tokens(), time.time()
    answer = httpx.post(f"{base}/chat", json=question, headers={"Origin": "http://localhost:5173"}, timeout=180)
    seconds, spent = time.time() - started, tokens() - before
    if answer.status_code != 200:
        return [f"/api/chat: HTTP {answer.status_code} {answer.text[:200]}"]
    body = answer.json()
    print(f"разговор: {seconds:.0f} с, токенов {spent}, отказ: {body['refused']}, запасной: {body['degraded']}, "
          f"источник: {body['citations'][0].get('article_title', '')[:60]!r}")
    if answer.headers.get("access-control-allow-origin") != "*":
        problems.append("/api/chat: браузер с другого адреса не получит ответ (нет заголовка CORS)")
    if set(body["answers"]) != set(ids) or body["degraded"]:
        problems.append("/api/chat: нет ответа всех экспертов или вернулся запасной текст")
    if not body["citations"] or not body["citations"][0].get("section_title"):
        problems.append("/api/chat: у ссылки нет названий статьи и раздела")
    before = tokens()
    other = httpx.post(f"{base}/chat", json={**question, "expert_id": ids[1]}, timeout=60).json()
    print(f"тот же вопрос другому эксперту: потрачено ещё токенов: {tokens() - before}")
    if tokens() != before or other["text"] != body["answers"][ids[1]]["text"]:
        problems.append("/api/chat: переключение эксперта потратило токены или дало другой текст")
    citation = body["citations"][0]
    source = httpx.get(f"{base}/sources/{citation['article_id']}/{citation['section_id']}", timeout=30)
    if source.status_code != 200 or not source.json().get("text"):
        problems.append("/api/sources: текст раздела не отдаётся")
    return problems


def check_voice(emulator_port: int) -> list[str]:
    """Голос насквозь: ответ озвучивается, запись распознаётся, повторное прослушивание не тратит бюджет."""
    base, problems = f"http://127.0.0.1:{APP_PORT}/api", []
    stats = lambda: httpx.get(f"http://127.0.0.1:{emulator_port}/_emulator/stats", timeout=10).json()  # noqa: E731
    phrase = "Да, чашка кофе после обеда может мешать уснуть."
    spoken = httpx.post(f"{base}/speak", json={"text": phrase, "expert_id": "nutritionist"},
                        headers={"Origin": "http://localhost:5173"}, timeout=120)
    if spoken.status_code != 200 or spoken.headers.get("content-type") != "audio/mpeg" or not spoken.content:
        return [f"/api/speak: HTTP {spoken.status_code} {spoken.text[:200]}"]
    if spoken.headers.get("access-control-allow-origin") != "*":
        problems.append("/api/speak: браузер с другого адреса не получит звук (нет заголовка CORS)")
    again = httpx.post(f"{base}/speak", json={"text": phrase, "expert_id": "nutritionist"}, timeout=120)
    heard = httpx.post(f"{base}/transcribe", content=spoken.content, headers={"Content-Type": "audio/mpeg"}, timeout=120)
    after = stats()
    print(f"голос: синтез {len(spoken.content)} байт, распознано: {heard.json().get('text', heard.text)[:60]!r}; "
          f"обращений к прокси: синтез {after['tts_calls']}, распознавание {after['stt_calls']}")
    if heard.status_code != 200 or heard.json().get("text") != phrase:
        problems.append(f"/api/transcribe: HTTP {heard.status_code}, текст не совпал с озвученным")
    if again.content != spoken.content or after["tts_calls"] != 1:
        problems.append("/api/speak: повторное прослушивание обратилось к прокси ещё раз")
    too_big = httpx.post(f"{base}/transcribe", content=b"\x00" * (5 * 1024 * 1024 + 1),
                         headers={"Content-Type": "audio/webm"}, timeout=120)
    if too_big.status_code != 413:
        problems.append(f"/api/transcribe: запись больше 5 МБ получила HTTP {too_big.status_code} вместо 413")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="data/stand-check-answers.json")
    parser.add_argument("--port", type=int, default=8900)
    parser.add_argument("--no-repeat", action="store_true", help="не проверять повторный /run")
    parser.add_argument("--no-chat", action="store_true", help="не проверять методы для клиента")
    args, emulator_args = parser.parse_known_args()
    code = next(a for a in emulator_args if "=" in a and not a.startswith("--")).split("=")[0]

    print("сборка образа…", flush=True)
    build_image()
    emulator = subprocess.Popen(
        [sys.executable, "-m", "scripts.emulator", "--port", str(args.port), *emulator_args], cwd=ROOT
    )
    problems, peak, stop = [], {"mib": 0.0}, threading.Event()
    try:
        for _ in range(60):
            if emulator.poll() is not None:
                sys.exit("эмулятор не запустился: проверьте ключи и свободен ли порт")
            try:
                httpx.get(f"http://127.0.0.1:{args.port}/health", timeout=2)
                break
            except httpx.HTTPError:
                time.sleep(0.5)
        start_container(args.port)
        print(f"сервер ответил на GET / через {wait_ready():.0f} с", flush=True)
        threading.Thread(target=watch_memory, args=(peak, stop), daemon=True).start()

        def run() -> tuple[httpx.Response, float]:
            started = time.time()
            response = httpx.post(
                f"http://127.0.0.1:{APP_PORT}/run", json={"case": 4, "dataset_code": code},
                headers={"Authorization": f"Bearer {RUN_TOKEN}"}, timeout=45 * 60,
            )
            return response, time.time() - started

        response, seconds = run()
        stats = httpx.get(f"http://127.0.0.1:{args.port}/_emulator/stats", timeout=10).json()
        print(f"/run: HTTP {response.status_code}, {seconds:.0f} с, {len(response.content)} байт, "
              f"{response.headers.get('content-type')}")
        if response.status_code != 200:
            problems.append(f"HTTP {response.status_code} вместо 200 — ноль за автоматическую часть")
        if "application/json" not in response.headers.get("content-type", ""):
            problems.append("Content-Type не application/json")
        if len(response.content) > MAX_BODY_BYTES:
            problems.append("ответ больше 32 МиБ")
        if seconds > 45 * 60:
            problems.append("дольше 45 минут")
        try:
            body = response.json()
            Path(args.out).write_text(json.dumps(body, ensure_ascii=False, indent=1), encoding="utf-8")
            verdict = httpx.post(f"http://127.0.0.1:{args.port}/case/4/validate", json=body,
                                 headers={"Authorization": f"Bearer {TEAM_TOKEN}"}, timeout=60).json()
            if not verdict["valid"]:
                problems.append(f"validate: {verdict['errors'][:3]}")
            answers = body.get("requests", [])
            fallbacks = sum("по ссылке" in a["text"] and len(a["text"].split()) < 35 for a in answers)
            print(f"ответов {len(answers)}, отказов {sum(a['refused'] for a in answers)}, запасных {fallbacks}")
        except (ValueError, KeyError, TypeError) as failure:
            problems.append(f"тело ответа не разбирается: {failure}")
        print(f"прокси: вызовов {stats['llm_calls']}, токенов {stats['tokens']} "
              f"(вход {stats['prompt_tokens']}, выход {stats['completion_tokens']}), "
              f"пик {stats['peak_requests_per_minute']} запросов в минуту, "
              f"429 по минуте {stats['http_429_minute']}, 429 по суткам {stats['http_429_daily']}, сбоев {stats['faults']}")

        if not args.no_repeat and response.status_code == 200:
            again, again_seconds = run()
            after = httpx.get(f"http://127.0.0.1:{args.port}/_emulator/stats", timeout=10).json()
            extra = after["tokens"] - stats["tokens"]
            print(f"повторный /run: HTTP {again.status_code}, {again_seconds:.1f} с, потрачено ещё токенов: {extra}")
            if again.status_code != 200 or again.content != response.content:
                problems.append("повторный /run вернул другой ответ")

        if not args.no_chat and response.status_code == 200:
            problems += check_client_methods(args.port)
            problems += check_voice(args.port)

        state = json.loads(sh("docker", "inspect", "--format", "{{json .State}}", CONTAINER).stdout)
        print(f"память: пик {peak['mib']:.0f} МиБ из 512; OOMKilled: {state['OOMKilled']}; работает: {state['Running']}")
        if state["OOMKilled"] or not state["Running"]:
            problems.append("контейнер упал")
        logs = sh("docker", "logs", CONTAINER, check=False)
        leaks = [word for word in (RUN_TOKEN, TEAM_TOKEN, LLM_KEY, code) if word in logs.stdout + logs.stderr]
        if leaks:
            problems.append(f"в логах решения есть то, чего там быть не должно: {leaks}")
    finally:
        stop.set()
        emulator.terminate()
        emulator.wait(timeout=20)  # порт должен освободиться до следующего запуска
        sh("docker", "rm", "-f", CONTAINER, check=False)
    print("ИТОГ:", "замечаний нет" if not problems else "; ".join(problems))
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
