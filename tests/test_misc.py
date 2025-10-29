import sys
import importlib
from types import SimpleNamespace
import textwrap
import pytest

from taranis_base_bot.misc import create_request_parser, get_model, get_hf_modelinfo


def test_create_request_parser_success():
    parser = create_request_parser({"text": "str"})
    data = {"text": "hello"}
    out = parser(data)
    assert out == {"text": "hello"}


def test_create_request_parser_missing_key():
    parser = create_request_parser({"text": "str"})
    with pytest.raises(ValueError) as ei:
        parser({"other": "x"})
    assert str(ei.value) == "Payload does not contain 'text' key!"


def test_create_request_parser_empty_value_string():
    parser = create_request_parser({"text": "str"})
    with pytest.raises(ValueError) as ei:
        parser({"text": ""})
    assert str(ei.value) == "No data provided for 'text' key!"


def test_create_request_parser_wrong_type():
    parser = create_request_parser({"text": "str"})
    with pytest.raises(ValueError) as ei:
        parser({"text": 123})
    assert str(ei.value) == "Data for 'text' is not of type 'str'"


def test_create_request_parser_empty_list_also_rejected():
    parser = create_request_parser({"items": list})
    with pytest.raises(ValueError) as ei:
        parser({"items": []})
    assert str(ei.value) == "No data provided for 'items' key!"


def _write_module(tmp_path, package_name: str, module_name: str, source: str):
    """Helper to write a package and a module file, returning package path."""
    pkg_dir = tmp_path / package_name
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / f"{module_name}.py").write_text(textwrap.dedent(source))
    return pkg_dir


def test_get_model_success(tmp_path, monkeypatch):
    """
    get_model should import <PACKAGE_NAME>.<MODEL>, derive the class name from MODEL,
    and return an instance of that class.
    """
    package = "dynpkg"
    module = "toy_model"
    src = """
    class ToyModel:
        def __init__(self):
            self.ok = True
    """
    _ = _write_module(tmp_path, package, module, src)

    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    cfg = SimpleNamespace(PACKAGE_NAME=package, MODEL=module)

    # Exercise
    model = get_model(cfg)
    assert getattr(model, "ok", False) is True

    for mod in [f"{package}.{module}", package]:
        sys.modules.pop(mod, None)


def test_get_model_missing_module_raises(tmp_path, monkeypatch):
    package = "dynpkg2"
    _write_module(tmp_path, package, "__init__", "")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    cfg = SimpleNamespace(PACKAGE_NAME=package, MODEL="does_not_exist")
    with pytest.raises(ImportError) as ei:
        get_model(cfg)
    assert f"Package {package} has no module named {cfg.MODEL}" in str(ei.value)

    for mod in [package]:
        sys.modules.pop(mod, None)


def test_get_model_missing_class_raises(tmp_path, monkeypatch):
    package = "dynpkg3"
    module = "bad_model"
    src = """
    class AnotherClass:
        pass
    """
    _write_module(tmp_path, package, module, src)
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()

    cfg = SimpleNamespace(PACKAGE_NAME=package, MODEL=module)
    with pytest.raises(ImportError) as ei:
        get_model(cfg)
    assert f"Module {module} has no class named BadModel" in str(ei.value)

    for mod in [f"{package}.{module}", package]:
        sys.modules.pop(mod, None)


def test_get_hf_modelinfo_http_error(requests_mock):
    model_name = "missing-model"
    requests_mock.get(
        f"https://huggingface.co/api/models/{model_name}",
        status_code=404,
        json={"error": "not found"},
    )
    out = get_hf_modelinfo(model_name)
    assert out["model"] == model_name
    assert "error" in out


def test_get_hf_modelinfo_exception_returns_fallback(monkeypatch):
    model_name = "any-model"

    import requests

    def unexpected_error(*a, **k):
        raise RuntimeError("Fail!")

    monkeypatch.setattr(requests, "get", unexpected_error)

    out = get_hf_modelinfo(model_name)
    assert out["model"] == model_name
    assert out["error"] == "Fail!"
