from flask import Flask
from typing import Callable, Any
from taranis_base_bot import blueprint
from taranis_base_bot.misc import get_model, get_hf_modelinfo, create_request_parser
from taranis_base_bot.decorators import api_key_required
from taranis_base_bot.blueprint import JSON
from taranis_base_bot.log import configure_logger
from taranis_base_bot.config import CommonSettings


def create_app(
    name: str,
    config: CommonSettings,
    url_prefix: str = "/",
    predict_fn: Callable[..., Any] | None = None,
    modelinfo_fn: Callable[[], Any] | None = None,
    request_parser: Callable[[JSON], dict[str, Any]] | None = None,
    method_decorators: list[Callable] | None = None,
    setup_logging: bool = True,
) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config)
    app.url_map.strict_slashes = False

    if setup_logging:
        configure_logger(
            module=config.PACKAGE_NAME,
            debug=config.DEBUG,
            colored=config.COLORED_LOGS,
            syslog_address=None,
        )

    model = None
    if predict_fn is None or modelinfo_fn is None:
        model = get_model(config)

    if predict_fn is None:
        predict_fn = model.predict  # type: ignore[assignment]

    if modelinfo_fn is None:

        def default_modelinfo_fn():
            if config.HF_MODEL_INFO:
                return get_hf_modelinfo(model.model_name)  # type: ignore[attr-defined]
            else:
                return model.model_name  # type: ignore[attr-defined]

        modelinfo_fn = default_modelinfo_fn

    if request_parser is None:
        request_parser = create_request_parser(config.PAYLOAD_KEY, str)

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


if __name__ == "__main__":
    raise SystemExit("Cannot run taranis-base-bot directly. Use 'from taranis-base-bot import create_app' instead.")
