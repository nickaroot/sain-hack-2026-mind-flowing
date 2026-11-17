"""Прогон решения в условиях стенда: контейнер с теми же ограничениями и эмулятор внешних сервисов.

Повторяет то, что делает стенд организаторов (по их инструкции и README архива команды):
1. собирает образ python:3.11-slim с пакетами из requirements.txt (сеть есть только при сборке);
2. запускает start.sh от пользователя 10001 с 512 МБ памяти, 1 CPU, 256 процессами, запись только в /tmp;
   имена сайтов не резолвятся — доступны только эмулятор сервиса хакатона и прокси;
3. каждые 2 секунды до 60 секунд ждёт любой HTTP-ответ на GET /;
4. один раз вызывает POST /run с токеном прогона и кодом набора и ждёт ответ до 45 минут;
5. проверяет то, за что автоматическая часть обнуляется: HTTP 200, JSON, размер, формат (validate).
Дополнительно: повторный /run не должен тратить лимит второй раз.

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="data/stand-check-answers.json")
    parser.add_argument("--port", type=int, default=8900)
    parser.add_argument("--no-repeat", action="store_true", help="не проверять повторный /run")
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

        state = json.loads(sh("docker", "inspect", "--format", "{{json .State}}", CONTAINER).stdout)
        print(f"память: пик {peak['mib']:.0f} МиБ из 512; OOMKilled: {state['OOMKilled']}; работает: {state['Running']}")
        if state["OOMKilled"] or not state["Running"]:
            problems.append("контейнер упал")
        leaks = [word for word in (RUN_TOKEN, TEAM_TOKEN, LLM_KEY, code)
                 if word in sh("docker", "logs", CONTAINER, check=False).stdout + sh("docker", "logs", CONTAINER, check=False).stderr]
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
