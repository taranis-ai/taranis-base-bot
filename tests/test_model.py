import sys
import types

import pytest


def test_get_model(customSettings, fake_pkg, fake_model):
    from taranis_base_bot.misc import get_model

    model = get_model(customSettings)
    assert isinstance(model, fake_model)


def test_get_model_missing_module(customSettings, monkeypatch):
    from taranis_base_bot.misc import get_model

    monkeypatch.delitem(sys.modules, customSettings.PACKAGE_NAME, raising=False)
    monkeypatch.delitem(sys.modules, f"{customSettings.PACKAGE_NAME}.{customSettings.MODEL}", raising=False)

    with pytest.raises(ImportError, match=f"Package {customSettings.PACKAGE_NAME} has no module named {customSettings.MODEL}"):
        get_model(customSettings)


def test_get_model_missing_class(customSettings, monkeypatch):
    from taranis_base_bot.misc import get_model

    pkg = types.ModuleType(customSettings.PACKAGE_NAME)
    pkg.__path__ = []

    mod = types.ModuleType(f"{customSettings.PACKAGE_NAME}.{customSettings.MODEL}")
    monkeypatch.setitem(sys.modules, customSettings.PACKAGE_NAME, pkg)
    monkeypatch.setitem(sys.modules, f"{customSettings.PACKAGE_NAME}.{customSettings.MODEL}", mod)

    with pytest.raises(
        ImportError,
        match=f"Module {customSettings.MODEL} has no class named FakeModel. Make sure the class exists in {customSettings.MODEL}.py",
    ):
        get_model(customSettings)
