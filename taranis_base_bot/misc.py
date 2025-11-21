from importlib import import_module
from pydoc import locate
from typing import Any, Callable

import requests

from taranis_base_bot.log import logger
from taranis_base_bot.protocols import Predictor


def create_request_parser(payload_schema: dict[str, dict]) -> Callable[[dict], dict]:
    def request_parser(data: dict) -> dict[str, Any]:
        unexpected_keys = set(data) - set(payload_schema)
        if unexpected_keys:
            logger.warning(f"The payload contains unexpected keys: {unexpected_keys}")

        accepted_data = {}
        for key, key_schema in payload_schema.items():
            if not key_schema.get("required", True) and key not in data:
                continue

            if key not in data:
                raise ValueError(f"Payload does not contain '{key}' key!")

            val = data.get(key)

            if val is None or not val:
                raise ValueError(f"No data provided for '{key}' key!")

            data_type = key_schema.get("type")

            if data_type and not isinstance(val, locate(data_type)):  # type: ignore
                raise ValueError(f"Data for '{key}' is not of type '{data_type}'")
            accepted_data[key] = data[key]
        return accepted_data

    return request_parser


def get_model(config) -> Predictor:
    package_name = config.PACKAGE_NAME
    model_name = config.MODEL
    if not package_name or not model_name:
        raise ValueError("Both PACKAGE_NAME and MODEL must be set in the configuration")
    try:
        module = import_module(f"{package_name}.{model_name}")
    except ModuleNotFoundError as e:
        raise ImportError(f"Package {package_name} has no module named {model_name}. Make sure the file {model_name}.py exists") from e

    class_name = "".join(part.capitalize() for part in model_name.split("_") if part)
    model_class = getattr(module, class_name, None)
    if model_class is None:
        raise ImportError(f"Module {model_name} has no class named {class_name}. Make sure the class exists in {model_name}.py")

    return model_class()


def get_hf_modelinfo(model_name: str) -> dict:
    """
    Fetch model metadata from Hugging Face.
    If anything fails, return a simple fallback.
    """
    try:
        url = f"https://huggingface.co/api/models/{model_name}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"model": model_name, "error": str(e)}
