from flask import Flask
from typing import Callable, Any
from taranis_base_bot import blueprint
from taranis_base_bot.config import get_settings
from taranis_base_bot.decorators import api_key_required
from taranis_base_bot.blueprint import JSON

def create_app(name: str, url_prefix: str, predict_fn: Callable[..., Any], modelinfo_fn: Callable[[], Any], request_parser: Callable[[JSON], dict[str, Any]], method_decorators: list[Callable]) -> Flask:
    app = Flask(name)
    app.config.from_object(get_settings())
    app.url_map.strict_slashes = False

    bp = blueprint.create_service_blueprint(
        name=name,
        url_prefix=url_prefix,
        predict_fn=predict_fn,
        modelinfo_fn=modelinfo_fn,
        request_parser=request_parser,
        method_decorators=method_decorators
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
        method_decorators=[api_key_required]
    )

if __name__ == "__main__":
    create_empty_app().run()
