import pytest
import respx
from taranis_base_bot.misc import create_request_parser, get_hf_modelinfo


def test_create_request_parser_success():
    parser = create_request_parser({"text": {"type": "str", "required": True}})
    data = {"text": "hello"}
    out = parser(data)
    assert out == {"text": "hello"}


def test_create_request_parser_multi_key():
    parser = create_request_parser({"text": {"type": "str", "required": True}, "other": {"type": "str", "required": True}})
    data = {"text": "hello", "other": "also hello"}
    out = parser(data)
    assert out == {"text": "hello", "other": "also hello"}


def test_create_request_parser_int_key():
    parser = create_request_parser({"number": {"type": "int", "required": True}})
    data = {"number": 2}
    out = parser(data)
    assert out == {"number": 2}


def test_create_request_parser_bool_key():
    parser = create_request_parser({"flag": {"type": "bool", "required": True}})
    data = {"flag": False}
    out = parser(data)
    assert out == {"flag": False}


def test_create_request_parser_optional_key_missing():
    parser = create_request_parser({"required_key": {"type": "str", "required": False}, "opt_key": {"type": "str", "required": False}})
    result = parser({"required_key": "hello"})
    assert result == {"required_key": "hello"}


def test_create_request_parser_optional_key_no_value():
    parser = create_request_parser({"opt_key": {"type": "str", "required": False}})
    with pytest.raises(ValueError) as ei:
        parser({"opt_key": None})
    assert str(ei.value) == "No data provided for 'opt_key' key!"


def test_create_request_parser_missing_key():
    parser = create_request_parser({"text": {"type": "str", "required": True}})
    with pytest.raises(ValueError) as ei:
        parser({"other": "x"})
    assert str(ei.value) == "Payload does not contain 'text' key!"


def test_create_request_parser_multi_key_missing_key():
    parser = create_request_parser({"text": {"type": "str", "required": True}, "other": {"type": "str", "required": True}})
    with pytest.raises(ValueError) as ei:
        parser({"text": "hello"})
    assert str(ei.value) == "Payload does not contain 'other' key!"


def test_create_request_parser_empty_value_string():
    parser = create_request_parser({"text": {"type": "str", "required": True}})
    with pytest.raises(ValueError) as ei:
        parser({"text": ""})
    assert str(ei.value) == "No data provided for 'text' key!"


def test_create_request_parser_wrong_type():
    parser = create_request_parser({"text": {"type": "str", "required": True}})
    with pytest.raises(ValueError) as ei:
        parser({"text": 123})
    assert str(ei.value) == "Data for 'text' is not of type 'str'"


def test_create_request_parser_empty_list_also_rejected():
    parser = create_request_parser({"items": {"type": "list", "required": True}})
    with pytest.raises(ValueError) as ei:
        parser({"items": []})
    assert str(ei.value) == "No data provided for 'items' key!"


@pytest.mark.asyncio
async def test_get_hf_modelinfo_http_error():
    model_name = "missing-model"

    with respx.mock(base_url="https://huggingface.co") as mock:
        mock.get(f"/api/models/{model_name}").respond(
            status_code=404,
            json={"error": "not found"},
        )

        out = await get_hf_modelinfo(model_name)

    assert out["model"] == model_name
    assert "error" in out


@pytest.mark.asyncio
async def test_get_hf_modelinfo_exception_returns_fallback():
    model_name = "any-model"

    with respx.mock(base_url="https://huggingface.co") as mock:
        mock.get(f"/api/models/{model_name}").mock(side_effect=RuntimeError("Fail!"))

        out = await get_hf_modelinfo(model_name)

    assert out["model"] == model_name
    assert out["error"] == "Fail!"
