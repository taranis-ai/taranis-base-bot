import pytest

from taranis_base_bot.config import get_settings

Config = get_settings()

@pytest.fixture
def client():
    from taranis_base_bot import create_app

    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def client_with_api_key(monkeypatch):
    """Client with API key authentication enabled"""
    # Temporarily set API key for testing
    monkeypatch.setenv("API_KEY", "test-api-key-123")

    # Reload config to pick up the new API key
    Config.API_KEY = "test-api-key-123"
    from taranis_base_bot import create_app

    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client
