from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    __version__: str
    __version_tuple__: tuple[str, ...]

from .helpers import *

if not TYPE_CHECKING:

    def __getattr__(name: str) -> Any:
        if name in ("__version__", "__version_tuple__"):
            from importlib import import_module

            try:
                module = import_module("._version", package=__package__)
            except ModuleNotFoundError:
                return "unknown"

            return getattr(module, name)

        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
