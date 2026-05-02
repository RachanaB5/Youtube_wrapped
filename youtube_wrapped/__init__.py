"""YouTube Wrapped: Watch history analysis package."""


def create_app():
    """
    Flask application factory.

    Imported lazily so ``import youtube_wrapped.analyst`` (or other light modules)
    does not pull in PyTorch / transformers.
    """
    from youtube_wrapped.routes import create_app as factory

    return factory()


__all__ = ["create_app"]
