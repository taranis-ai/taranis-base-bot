from taranis_base_bot import create_app
from taranis_base_bot.config import get_common_settings


def test_default_modelinfo_uses_hf_when_enabled(monkeypatch):
    import taranis_base_bot as pkg

    class DummyModel:
        model_name = "hf-model"
        modelinfo = {"should_not_be_used": True}

        def predict(self, **kw):
            return kw

    def fake_get_model(_cfg):
        return DummyModel()

    def fake_get_hf_modelinfo(name):
        assert name == "hf-model"
        return {"source": "hf", "name": name}

    monkeypatch.setattr(pkg, "get_model", fake_get_model)
    monkeypatch.setattr(pkg, "get_hf_modelinfo", fake_get_hf_modelinfo)

    cfg = get_common_settings()
    cfg.HF_MODEL_INFO = True

    app = create_app(
        name="svc-hf",
        config=cfg,
        request_parser=lambda x: x,
        method_decorators=[],
    )
    with app.test_client() as c:
        r = c.get("/modelinfo")
        assert r.status_code == 200
        assert r.get_json() == {"source": "hf", "name": "hf-model"}


def test_default_modelinfo_uses_local_when_disabled(monkeypatch):
    import taranis_base_bot as pkg

    class DummyModel:
        model_name = "hf-model"
        modelinfo = {"source": "local"}

        def predict(self, **kw):
            return kw

    def fake_get_model(_cfg):
        return DummyModel()

    def fake_get_hf_modelinfo(_name):
        raise AssertionError("HF modelinfo should not be called")

    monkeypatch.setattr(pkg, "get_model", fake_get_model)
    monkeypatch.setattr(pkg, "get_hf_modelinfo", fake_get_hf_modelinfo)

    cfg = get_common_settings()
    cfg.HF_MODEL_INFO = False

    app = create_app(
        name="svc-local",
        config=cfg,
        request_parser=lambda x: x,
        method_decorators=[],
    )
    with app.test_client() as c:
        r = c.get("/modelinfo")
        assert r.status_code == 200
        assert r.get_json() == {"source": "local"}


def test_default_request_parser_uses_config_payload_key(monkeypatch):
    import taranis_base_bot as pkg

    class DummyModel:
        model_name = "m"
        modelinfo = {"m": True}

        def predict(self, **kw):
            return kw

    monkeypatch.setattr(pkg, "get_model", lambda _cfg: DummyModel())

    cfg = get_common_settings()
    cfg.PAYLOAD_KEY = "test"

    app = create_app(
        name="svc-parser",
        config=cfg,
        method_decorators=[],
    )
    with app.test_client() as c:
        r = c.post("/", json={"test": "test val"})
        assert r.status_code == 200
        assert r.get_json() == {"test": "test val"}


def test_url_prefix_routes(monkeypatch):
    import taranis_base_bot as pkg

    class DummyModel:
        model_name = "m"
        modelinfo = {"mi": True}

        def predict(self, **kw):
            return kw

    monkeypatch.setattr(pkg, "get_model", lambda _cfg: DummyModel())

    cfg = get_common_settings()
    cfg.PAYLOAD_KEY = "text"
    app = create_app(
        name="svc-prefixed",
        config=cfg,
        url_prefix="/v1",
        method_decorators=[],
    )

    with app.test_client() as c:
        r1 = c.get("/v1/health")
        assert r1.status_code == 200
        assert r1.get_json() == {"status": "ok"}

        r2 = c.get("/v1/modelinfo")
        assert r2.status_code == 200
        assert r2.get_json() == {"mi": True}

        r3 = c.post("/v1/", json={"text": "hello"})
        assert r3.status_code == 200
        assert r3.get_json() == {"text": "hello"}
