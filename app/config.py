"""Настройки из окружения. На стенде переменные уже заданы организаторами."""

import os

CASE_ID = 4

HACK_API_URL = os.environ.get("HACK_API_URL", "https://hack-api.be-healthy.tech").rstrip("/")
HACK_TEAM_TOKEN = os.environ.get("HACK_TEAM_TOKEN", "")

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "")

# Набор до ~20 МБ и медленный стенд: читаем терпеливо, но не бесконечно.
HACK_API_TIMEOUT_SECONDS = 120.0

# Суточный лимит прокси — 300 тыс. токенов на команду, и исчерпать его нельзя: остаток прогона получил бы
# запасные ответы. Когда прогон потратил больше этого числа, повторные попытки больше не делаются.
RETRY_TOKEN_BUDGET = int(os.environ.get("RETRY_TOKEN_BUDGET", "230000"))
# Список названий статей уходит в нейросеть целиком; для очень большой базы он занял бы слишком много токенов.
CATALOG_MAX_ARTICLES = 300
