from typing import Any, Callable

from flask import Flask


def create_app(
    name: str,
    config,
    url_prefix: str = "/",
    predict_fn: Callable[..., Any] | None = None,
    modelinfo_fn: Callable[[], Any] | None = None,
    request_parser: Callable[[Any], dict[str, Any]] | None = None,
    method_decorators: list[Callable] | None = None,
) -> Flask:
    from taranis_base_bot.setup import setup

    return setup(
        name=name,
        config=config,
        url_prefix=url_prefix,
        predict_fn=predict_fn,
        modelinfo_fn=modelinfo_fn,
        request_parser=request_parser,
        method_decorators=method_decorators,
    )


if __name__ == "__main__":
    raise SystemExit("Cannot run taranis-base-bot directly. Use 'from taranis-base-bot import create_app' instead.")
