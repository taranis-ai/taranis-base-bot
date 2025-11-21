from typing import Any, Callable

from flask import Flask

from taranis_base_bot import blueprint
from taranis_base_bot.decorators import api_key_required
from taranis_base_bot.log import logger
from taranis_base_bot.misc import create_request_parser, get_hf_modelinfo, get_model


def setup(
    name: str,
    config,
    url_prefix: str = "/",
    predict_fn: Callable[..., Any] | None = None,
    modelinfo_fn: Callable[[], Any] | None = None,
    request_parser: Callable[[Any], dict[str, Any]] | None = None,
    method_decorators: list[Callable] | None = None,
) -> Flask:
    logger.info(f"Creating app for Bot {config.PACKAGE_NAME} with MODEL {config.MODEL}")
    app = Flask(__name__)
    app.config.from_object(config)
    app.url_map.strict_slashes = False
    logger.reconfigure_from_settings(config)

    if predict_fn is None or modelinfo_fn is None:
        model = get_model(config)

        if predict_fn is None:
            predict_fn = model.predict

        if modelinfo_fn is None:

            def default_modelinfo_fn():
                if config.HF_MODEL_INFO:
                    return get_hf_modelinfo(model.model_name)  # type: ignore[attr-defined]
                else:
                    return model.model_name

            modelinfo_fn = default_modelinfo_fn

    if request_parser is None:
        request_parser = create_request_parser(config.PAYLOAD_SCHEMA)

    if method_decorators is None:
        method_decorators = [api_key_required]

    bp = blueprint.create_service_blueprint(
        name=name,
        url_prefix=url_prefix,
        predict_fn=predict_fn,
        modelinfo_fn=modelinfo_fn,
        request_parser=request_parser,
        method_decorators=method_decorators,
    )
    app.register_blueprint(bp)
    return app
