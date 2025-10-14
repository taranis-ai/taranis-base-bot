from flask import Flask
from typing import Callable, Any
from taranis_base_bot import blueprint
from taranis_base_bot.config import get_settings
from taranis_base_bot.decorators import api_key_required
from taranis_base_bot.blueprint import JSON
from taranis_base_bot.log import configure_logger


def create_app(
    name: str,
    url_prefix: str,
    predict_fn: Callable[..., Any],
    modelinfo_fn: Callable[[], Any],
    request_parser: Callable[[JSON], dict[str, Any]],
    method_decorators: list[Callable],
) -> Flask:
    app = Flask(name)
    config = get_settings()
    app.config.from_object(config)
    app.url_map.strict_slashes = False

    configure_logger(
        module=config.MODULE_ID,
        debug=config.DEBUG,
        colored=config.COLORED_LOGS,
        syslog_address=None,
    )

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


def create_empty_app() -> Flask:
    return create_app(
        name="taranis-base-bot",
        url_prefix="/",
        predict_fn=lambda x: x,
        modelinfo_fn=lambda: {"model": "test"},
        request_parser=lambda x: x,
        method_decorators=[api_key_required],
    )


if __name__ == "__main__":
    raise SystemExit("Cannot run taranis-base-bot directly. Use 'from taranis-base-bot import create_app' instead.")
