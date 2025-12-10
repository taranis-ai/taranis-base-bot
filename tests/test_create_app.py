import pytest
from taranis_base_bot import create_app


@pytest.mark.asyncio
async def test_default_modelinfo_uses_hf_when_enabled(custom_settings, fake_pkg, hf_modelinfo_mock):
    app = create_app(
        name="svc-hf",
        config=custom_settings,
    )
    async with app.test_client() as c:
        r = await c.get("/modelinfo")
        assert r.status_code == 200
        data = await r.get_json()
        assert data == hf_modelinfo_mock


@pytest.mark.asyncio
async def test_default_modelinfo_uses_name_when_disabled(custom_settings, fake_pkg):
    custom_settings.HF_MODEL_INFO = False

    app = create_app(
        name="svc-local",
        config=custom_settings,
    )
    async with app.test_client() as c:
        r = await c.get("/modelinfo")
        assert r.status_code == 200
        data = await r.get_json()
        assert data == custom_settings.MODEL


@pytest.mark.asyncio
async def test_default_request_parser_uses_config_payload_schema(custom_settings, fake_pkg):
    custom_settings.PAYLOAD_SCHEMA = {"test": {"type": "str", "required": True}}

    app = create_app(name="svc-parser", config=custom_settings, method_decorators=[])
    async with app.test_client() as c:
        r = await c.post("/", json={"test": "test val"})
        assert r.status_code == 200
        data = await r.get_json()
        assert data == {"test": "test val"}


@pytest.mark.asyncio
async def test_url_prefix_routes(custom_settings, fake_pkg, hf_modelinfo_mock):
    custom_settings.PAYLOAD_SCHEMA = {"text": {"type": "str", "required": True}}

    app = create_app(
        name="svc-prefixed",
        config=custom_settings,
        url_prefix="/v1",
        method_decorators=[],
    )

    async with app.test_client() as c:
        r1 = await c.get("/v1/health")
        assert r1.status_code == 200
        data = await r1.get_json()
        assert data == {"status": "ok"}

        r2 = await c.get("/v1/modelinfo")
        assert r2.status_code == 200
        data = await r2.get_json()
        assert data == hf_modelinfo_mock

        r3 = await c.post("/v1/", json={"text": "hello"})
        assert r3.status_code == 200
        data = await r3.get_json()
        assert data == {"text": "hello"}
