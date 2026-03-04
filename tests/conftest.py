import os
import sys
import types
from typing import Literal

import pytest
import pytest_asyncio
import respx
from dotenv import load_dotenv

from taranis_base_bot.config import CommonSettings
from taranis_base_bot.misc import get_hf_modelinfo

base_dir = os.path.dirname(os.path.abspath(__file__))
env_file = os.path.join(base_dir, ".env.test")
current_path = os.getcwd()

load_dotenv(dotenv_path=env_file, override=True)


@pytest.fixture
def custom_settings():
    class CustomSettings(CommonSettings):
        MODEL: Literal["fake_model", "super", "good"] = "fake_model"
        PACKAGE_NAME: str = "fake_bot"
        HF_MODEL_INFO: bool = True

    yield CustomSettings()


@pytest_asyncio.fixture
async def client(custom_settings):
    from taranis_base_bot import create_app

    async def predict_fn(**kwargs):
        return kwargs

    app = create_app(
        name="taranis_base_bot",
        config=custom_settings,
        predict_fn=predict_fn,
        modelinfo_fn=lambda: get_hf_modelinfo(custom_settings.MODEL),
        request_parser=lambda x: x,
    )
    app.config["TESTING"] = True
    async with app.test_client() as client:
        yield client


@pytest_asyncio.fixture
async def client_with_request_parser(custom_settings):
    from taranis_base_bot import create_app

    def request_parser(data: dict[str, str]) -> dict[str, str]:
        if "text" not in data:
            raise ValueError("Could not parse payload. Check bot logs for more details.")
        return {"text": data["text"]}

    async def predict_fn(**kwargs):
        return kwargs

    app = create_app(
        name="taranis_base_bot",
        config=custom_settings,
        predict_fn=predict_fn,
        modelinfo_fn=lambda: get_hf_modelinfo(custom_settings.MODEL),
        request_parser=request_parser,
    )
    app.config["TESTING"] = True
    async with app.test_client() as client:
        yield client


@pytest_asyncio.fixture
async def client_with_predict(custom_settings):
    from taranis_base_bot import create_app

    async def predict_func(text: str) -> dict[str, int]:
        if not isinstance(text, str):
            raise ValueError("Bot execution failed. Check bot logs for more details.")
        return {"len": len(text)}

    app = create_app(
        name="taranis_base_bot",
        config=custom_settings,
        predict_fn=predict_func,
        modelinfo_fn=lambda: get_hf_modelinfo(custom_settings.MODEL),
        request_parser=lambda x: x,
    )
    app.config["TESTING"] = True
    async with app.test_client() as client:
        yield client


@pytest_asyncio.fixture
async def client_with_modelinfo_fn(custom_settings):
    from taranis_base_bot import create_app

    async def predict_fn(**kwargs):
        return kwargs

    app = create_app(
        name="taranis_base_bot", config=custom_settings, predict_fn=predict_fn, modelinfo_fn=lambda: get_hf_modelinfo(custom_settings.MODEL)
    )
    app.config["TESTING"] = True
    async with app.test_client() as client:
        yield client


@pytest_asyncio.fixture
async def client_with_api_key(custom_settings):
    """Client with API key authentication enabled"""
    from taranis_base_bot import create_app

    custom_settings.API_KEY = "test-api-key-123"

    async def predict_fn(**kwargs):
        return kwargs

    app = create_app(
        name="taranis_base_bot_with_api_key",
        config=custom_settings,
        predict_fn=predict_fn,
        modelinfo_fn=lambda: get_hf_modelinfo(custom_settings.MODEL),
        request_parser=lambda x: x,
    )
    app.config["TESTING"] = True
    async with app.test_client() as client:
        yield client


@pytest.fixture
def fake_model(custom_settings):
    class FakeModel:
        model_name = custom_settings.MODEL

        async def predict(self, **kwargs):
            return {**kwargs}

    yield FakeModel


@pytest.fixture
def hf_modelinfo_mock(custom_settings):
    with respx.mock(base_url="https://huggingface.co") as mock:
        mock.get(f"/api/models/{custom_settings.MODEL}").respond(
            status_code=200,
            json={"model": custom_settings.MODEL},
        )
        yield {"model": custom_settings.MODEL}


@pytest.fixture
def fake_pkg(monkeypatch, fake_model, custom_settings):
    pkg = types.ModuleType(custom_settings.PACKAGE_NAME)
    pkg.__path__ = []

    mod = types.ModuleType(f"{custom_settings.PACKAGE_NAME}.{custom_settings.MODEL}")

    mod.FakeModel = fake_model  # type: ignore

    monkeypatch.setitem(sys.modules, custom_settings.PACKAGE_NAME, pkg)
    monkeypatch.setitem(sys.modules, f"{custom_settings.PACKAGE_NAME}.{custom_settings.MODEL}", mod)

    yield pkg, mod
