from collections.abc import Callable
from functools import wraps
from traceback import TracebackException
from typing import Any

from jetpytools import CustomError, FuncExcept, SupportsString


class PluginNotFoundError(CustomError, AttributeError):
    def __init__(
        self,
        message: SupportsString | None = None,
        name: str | None = None,
        func: FuncExcept | None = None,
        reason: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, func, reason, **kwargs)
        AttributeError.__init__(self, message, name=name)


class FunctionNotFoundError(CustomError, AttributeError):
    def __init__(
        self,
        message: SupportsString | None = None,
        name: str | None = None,
        func: FuncExcept | None = None,
        reason: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message, func, reason, **kwargs)
        AttributeError.__init__(self, message, name=name)


def plugin_guard[**P, R](getplugin_func: Callable[P, R]) -> Callable[P, R]:

    @wraps(getplugin_func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return getplugin_func(*args, **kwargs)
        except AttributeError as e:
            if (
                TracebackException.from_exception(e).stack[-1].filename.endswith("vapoursynth.pyx")
                and e.name
                and "plugin namespace" in str(e)
            ):
                raise PluginNotFoundError(
                    f"Could not find plugin '{e.name}'. Please check your spelling or installation.",
                    e.name,
                ) from e

            raise

    return wrapper


def function_guard[**P, R](getfunction_func: Callable[P, R]) -> Callable[P, R]:
    @wraps(getfunction_func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return getfunction_func(*args, **kwargs)
        except AttributeError as e:
            if (
                TracebackException.from_exception(e).stack[-1].filename.endswith("vapoursynth.pyx")
                and e.name
                and "no function named" in str(e)
            ):
                raise FunctionNotFoundError(f"Could not find function '{e.name}'", e.name) from e

            raise

    return wrapper
