from __future__ import annotations

from copy import copy
from inspect import isclass, ismethod
from logging import INFO, Handler, LogRecord, basicConfig
from typing import Any, Iterable, Mapping

from jetpytools import norm_func_name
from rich.console import Console
from rich.logging import RichHandler

from ._modules import JETPACK_MODULES

__all__ = ["setup_logging"]


def setup_logging(level: str | int = INFO, handlers: Iterable[Handler] | None = None, **kwargs: Any) -> None:
    """
    Configure global logging.

    Args:
        level: Log level. Defaults to INFO.
        handlers: "None" will add a custom rich-based handler, with custom formatting for certain values.
        kwargs: Arguments forwarded to logging.basicConfig
    """
    if handlers is None:
        handlers = [CustomJetHandler(console=Console(stderr=True), show_time=False, rich_tracebacks=True)]

    kwargs = {"format": "{asctime}: {name}: {message}", "style": "{"} | kwargs

    basicConfig(level=level, handlers=handlers, **kwargs)


class CustomJetHandler(RichHandler):
    def format(self, record: LogRecord) -> str:
        # Return a modified shallow copy of the LogRecord with transformed
        # parameters for specific loggers.
        if record.name.startswith(JETPACK_MODULES):
            record = copy(record)
            if isinstance(record.args, tuple):
                # Convert tuple -> dict -> transform -> back to tuple
                transformed = _transform_record_args(dict(enumerate(record.args)))
                record.args = tuple(transformed.values())
            elif isinstance(record.args, Mapping):
                # Transform a dict-like args mapping
                record.args = _transform_record_args(record.args)

        return super().format(record)


def _transform_record_args[T](args: Mapping[T, object]) -> dict[T, object]:
    """
    Transform values in the args dictionary based on type.
    """
    transformed = dict[T, object]()

    for key, value in args.items():
        new_value = value

        # Special formatting for vapoursyth objects
        if getattr(value, "__module__", "") == "vapoursynth":
            value_t = type(value) if not isclass(value) else value

            if value_t.__str__ is not object.__str__:
                new_value = repr(value)

        # Normalize method names
        elif ismethod(value):
            new_value = norm_func_name(value)

        transformed[key] = new_value

    return transformed
