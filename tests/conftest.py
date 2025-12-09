import os
import sys
import types
from typing import Literal

import pytest_asyncio
import respx
from dotenv import load_dotenv

from taranis_base_bot.config import CommonSettings
from taranis_base_bot.misc import get_hf_modelinfo

base_dir = os.path.dirname(os.path.abspath(__file__))
env_file = os.path.join(base_dir, ".env.test")
current_path = os.getcwd()

load_dotenv(dotenv_path=env_file, override=True)


@pytest_asyncio.fixture
def customSettings():
    class CustomSettings(CommonSettings):
        MODEL: Literal["fake_model", "super", "good"] = "fake_model"
        PACKAGE_NAME: str = "fake_bot"
        HF_MODEL_INFO: bool = True

    yield CustomSettings()


@pytest_asyncio.fixture
async def client(customSettings):
    from taranis_base_bot import create_app

    async def predict_fn(**kwargs):
        return kwargs

    app = create_app(
        name="taranis_base_bot",
        config=customSettings,
        predict_fn=predict_fn,
        modelinfo_fn=lambda: get_hf_modelinfo(customSettings.MODEL),
        request_parser=lambda x: x,
    )
    app.config["TESTING"] = True
    async with app.test_client() as client:
        yield client


@pytest_asyncio.fixture
async def client_with_request_parser(customSettings):
    from taranis_base_bot import create_app

    def request_parser(data: dict[str, str]) -> dict[str, str]:
        if "text" not in data:
            raise ValueError("Could not parse payload. Check bot logs for more details.")
        return {"text": data["text"]}

    async def predict_fn(**kwargs):
        return kwargs

    app = create_app(
        name="taranis_base_bot",
        config=customSettings,
        predict_fn=predict_fn,
        modelinfo_fn=lambda: get_hf_modelinfo(customSettings.MODEL),
        request_parser=request_parser,
    )
    app.config["TESTING"] = True
    async with app.test_client() as client:
        yield client


@pytest_asyncio.fixture
async def client_with_predict(customSettings):
    from taranis_base_bot import create_app

    async def predict_func(text: str) -> dict[str, int]:
        if not isinstance(text, str):
            raise ValueError("Bot execution failed. Check bot logs for more details.")
        return {"len": len(text)}

    app = create_app(
        name="taranis_base_bot",
        config=customSettings,
        predict_fn=predict_func,
        modelinfo_fn=lambda: get_hf_modelinfo(customSettings.MODEL),
        request_parser=lambda x: x,
    )
    app.config["TESTING"] = True
    async with app.test_client() as client:
        yield client


@pytest_asyncio.fixture
async def client_with_modelinfo_fn(customSettings):
    from taranis_base_bot import create_app

    async def predict_fn(**kwargs):
        return kwargs

    app = create_app(
        name="taranis_base_bot", config=customSettings, predict_fn=predict_fn, modelinfo_fn=lambda: get_hf_modelinfo(customSettings.MODEL)
    )
    app.config["TESTING"] = True
    async with app.test_client() as client:
        yield client


@pytest_asyncio.fixture
async def client_with_api_key(customSettings):
    """Client with API key authentication enabled"""
    from taranis_base_bot import create_app

    customSettings.API_KEY = "test-api-key-123"

    async def predict_fn(**kwargs):
        return kwargs

    app = create_app(
        name="taranis_base_bot_with_api_key",
        config=customSettings,
        predict_fn=predict_fn,
        modelinfo_fn=lambda: get_hf_modelinfo(customSettings.MODEL),
        request_parser=lambda x: x,
    )
    app.config["TESTING"] = True
    async with app.test_client() as client:
        yield client


@pytest_asyncio.fixture
def fake_model(customSettings):
    class FakeModel:
        model_name = customSettings.MODEL

        async def predict(self, **kwargs):
            return {**kwargs}

    yield FakeModel


@pytest_asyncio.fixture
def hf_modelinfo_mock(customSettings):
    result = {"id": f"{customSettings.MODEL}", "private": False, "tags": [], "downloads": 0}
    with respx.mock(base_url="https://huggingface.co") as mock:
        mock.get(f"/api/models/{customSettings.MODEL}").respond(
            status_code=200,
            json=result,
        )
        yield result


@pytest_asyncio.fixture
def fake_pkg(monkeypatch, fake_model, customSettings):
    pkg = types.ModuleType(customSettings.PACKAGE_NAME)
    pkg.__path__ = []

    mod = types.ModuleType(f"{customSettings.PACKAGE_NAME}.{customSettings.MODEL}")

    mod.FakeModel = fake_model  # type: ignore

    monkeypatch.setitem(sys.modules, customSettings.PACKAGE_NAME, pkg)
    monkeypatch.setitem(sys.modules, f"{customSettings.PACKAGE_NAME}.{customSettings.MODEL}", mod)

    yield pkg, mod
