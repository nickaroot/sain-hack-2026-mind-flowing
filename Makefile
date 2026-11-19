# Короткие команды разработки. На стенде не используется: он ставит requirements.txt и запускает start.sh.
.PHONY: install lint types test check run stand

install:
	pip install -r requirements-dev.txt

lint:
	ruff check app scripts tests

types:
	mypy

test:
	pytest -q --cov

# То же, что проверяет GitHub Actions перед деплоем на стенд.
check: lint types test

run:
	bash start.sh

# Решение в контейнере с ограничениями стенда и заглушкой сервисов организаторов (нужны Docker и data/).
stand:
	python -m scripts.stand_check --dataset case4-dev-v2=data/case4-dev-v2.json --cache emu-dev
