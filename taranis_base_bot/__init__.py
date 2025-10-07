from flask import Flask
from taranis_base_bot import blueprint
from taranis_base_bot.config import get_settings
from taranis_base_bot.decorators import api_key_required


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(get_settings())
    app.url_map.strict_slashes = False

    bp = blueprint.create_service_blueprint(
        name="taranis-base-bot",
        url_prefix="/",
        predict_fn=lambda x: x,
        modelinfo_provider=lambda: {"model": "test"},
        request_parser=lambda x: x,
        method_decorators=[api_key_required]
    )
    app.register_blueprint(bp)
    return app


def init(app: Flask) -> None:
    bp = blueprint.create_service_blueprint(
        name="taranis-base-bot",
        url_prefix="/",
        predict_fn=lambda x: x,
        modelinfo_provider=lambda: {"model": "test"},
        request_parser=lambda x: x,
        method_decorators=[api_key_required]
    )
    app.register_blueprint(bp)


if __name__ == "__main__":
    create_app().run()
