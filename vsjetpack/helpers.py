import sys
from collections.abc import Callable, Iterable, Mapping
from copy import copy
from functools import wraps
from inspect import getmodule, isclass
from logging import INFO, Handler, LogRecord, basicConfig
from types import ModuleType
from typing import Any, Literal

from jetpytools import DependencyNotFoundError, norm_func_name

__all__ = ["is_from_vs_module", "lazy_load", "require_jet_dependency", "setup_logging"]


_vs_module: ModuleType | None = None


def is_from_vs_module(obj: Any) -> bool:
    """Returns true if the module in which the obj was defined is VapourSynth."""

    global _vs_module

    if not _vs_module:
        import vapoursynth

        _vs_module = vapoursynth

    if hasattr(obj, "__module__"):
        return sys.modules[obj.__module__] is _vs_module

    return getmodule(type(obj)) is _vs_module


def lazy_load(name: str, package: str | None = None, exc: Callable[[], Exception] | None = None) -> ModuleType:
    """Lazily load a package."""

    if name in sys.modules:
        return sys.modules[name]

    from importlib.util import LazyLoader, find_spec, module_from_spec

    spec = find_spec(name, package)

    if spec is None:
        raise exc() if exc else ModuleNotFoundError(f"No module named {name!r}", name=name, path=__file__)

    module = module_from_spec(spec)

    if spec.loader is None:
        raise exc() if exc else NotImplementedError

    loader = LazyLoader(spec.loader)
    loader.exec_module(module)

    return module


def require_jet_dependency[**P, R](
    *name: Literal["scipy", "rich", "psutil"],
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator that raises DependencyNotFoundError when a specific package is missing."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                return func(*args, **kwargs)
            except ImportError as e:
                if e.name in name:
                    exc = DependencyNotFoundError(
                        func,
                        e.name,
                        "Missing dependency '{package}' for function '{func_name}'. Please install vsjetpack[full]",
                        func_name=func,
                    )

                    raise exc from None
            raise

        return wrapper

    return decorator


_JETPACK_MODULES = (
    "vsaa",
    "vsdeband",
    "vsdehalo",
    "vsdeinterlace",
    "vsdenoise",
    "vsexprtools",
    "vsjetpack",
    "vskernels",
    "vsmasktools",
    "vsrgtools",
    "vsscale",
    "vssource",
    "vstools",
)


@require_jet_dependency("rich")
def setup_logging(level: str | int = INFO, handlers: Iterable[Handler] | None = None, **kwargs: Any) -> None:
    """
    Configure global logging.

    Args:
        level: Log level. Defaults to INFO.
        handlers: "None" will add a custom rich-based handler, with custom formatting for certain values.
        kwargs: Arguments forwarded to logging.basicConfig
    """
    if handlers is None:
        from rich.console import Console
        from rich.logging import RichHandler
        from rich.text import Text

        class CustomJetHandler(RichHandler):
            def format(self, record: LogRecord) -> str:
                # Return a modified shallow copy of the LogRecord with transformed
                # parameters for specific loggers.
                if record.name.startswith(_JETPACK_MODULES):
                    record = copy(record)
                    if isinstance(record.args, tuple):
                        transformed = _transform_record_args(dict(enumerate(record.args)))
                        record.args = tuple(transformed.values())
                    elif isinstance(record.args, Mapping):
                        record.args = _transform_record_args(record.args)

                return super().format(record)

        handlers = [
            CustomJetHandler(
                console=Console(stderr=True),
                omit_repeated_times=False,
                show_time=True,
                rich_tracebacks=True,
                log_time_format=lambda dt: Text("[{}.{:03d}]".format(dt.strftime("%H:%M:%S"), dt.microsecond // 1000)),
            )
        ]

    kwargs = {"format": "{name}: {message}", "style": "{"} | kwargs

    basicConfig(level=level, handlers=handlers, **kwargs)


def _transform_record_args[T](args: Mapping[T, object]) -> dict[T, object]:
    """
    Transform values in the args dictionary based on type.
    """
    transformed = dict[T, object]()

    for key, value in args.items():
        new_value = value

        # Normalize method and class names
        if callable(value) or isclass(value):
            new_value = norm_func_name(value)

        transformed[key] = new_value

    return transformed
