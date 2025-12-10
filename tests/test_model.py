import sys
import types

import pytest


def test_get_model(custom_settings, fake_pkg, fake_model):
    from taranis_base_bot.misc import get_model

    model = get_model(custom_settings)
    assert isinstance(model, fake_model)


def test_get_model_missing_module(custom_settings, monkeypatch):
    from taranis_base_bot.misc import get_model

    monkeypatch.delitem(sys.modules, custom_settings.PACKAGE_NAME, raising=False)
    monkeypatch.delitem(sys.modules, f"{custom_settings.PACKAGE_NAME}.{custom_settings.MODEL}", raising=False)

    with pytest.raises(ImportError, match=f"Package {custom_settings.PACKAGE_NAME} has no module named {custom_settings.MODEL}"):
        get_model(custom_settings)


def test_get_model_missing_class(custom_settings, monkeypatch):
    from taranis_base_bot.misc import get_model

    pkg = types.ModuleType(custom_settings.PACKAGE_NAME)
    pkg.__path__ = []

    mod = types.ModuleType(f"{custom_settings.PACKAGE_NAME}.{custom_settings.MODEL}")
    monkeypatch.setitem(sys.modules, custom_settings.PACKAGE_NAME, pkg)
    monkeypatch.setitem(sys.modules, f"{custom_settings.PACKAGE_NAME}.{custom_settings.MODEL}", mod)

    with pytest.raises(
        ImportError,
        match=f"Module {custom_settings.MODEL} has no class named FakeModel. Make sure the class exists in {custom_settings.MODEL}.py",
    ):
        get_model(custom_settings)
