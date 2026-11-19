"""Общее для всех тестов: приложение одно на процесс, поэтому его память очищается перед каждым тестом."""

import pytest

from app.main import app


@pytest.fixture(autouse=True)
def fresh_response_cache():
    app.state.responses.clear()  # иначе ответ заглушки нейросети из одного теста достался бы другому
