from taranis_base_bot import create_app


def test_default_modelinfo_uses_hf_when_enabled(customSettings, fake_pkg, hf_modelinfo_mock):
    app = create_app(
        name="svc-hf",
        config=customSettings,
    )
    with app.test_client() as c:
        r = c.get("/modelinfo")
        assert r.status_code == 200
        assert r.get_json() == hf_modelinfo_mock


def test_default_modelinfo_uses_name_when_disabled(customSettings, fake_pkg):
    customSettings.HF_MODEL_INFO = False

    app = create_app(
        name="svc-local",
        config=customSettings,
    )
    with app.test_client() as c:
        r = c.get("/modelinfo")
        assert r.status_code == 200
        assert r.get_json() == customSettings.MODEL


def test_default_request_parser_uses_config_payload_schema(customSettings, fake_pkg):
    customSettings.PAYLOAD_SCHEMA = {"test": {"type": "str", "required": True}}

    app = create_app(name="svc-parser", config=customSettings, method_decorators=[])
    with app.test_client() as c:
        r = c.post("/", json={"test": "test val"})
        assert r.status_code == 200
        assert r.get_json() == {"test": "test val"}


def test_url_prefix_routes(customSettings, fake_pkg, hf_modelinfo_mock):
    customSettings.PAYLOAD_SCHEMA = {"text": {"type": "str", "required": True}}

    app = create_app(
        name="svc-prefixed",
        config=customSettings,
        url_prefix="/v1",
        method_decorators=[],
    )

    with app.test_client() as c:
        r1 = c.get("/v1/health")
        assert r1.status_code == 200
        assert r1.get_json() == {"status": "ok"}

        r2 = c.get("/v1/modelinfo")
        assert r2.status_code == 200
        assert r2.get_json() == hf_modelinfo_mock

        r3 = c.post("/v1/", json={"text": "hello"})
        assert r3.status_code == 200
        assert r3.get_json() == {"text": "hello"}
