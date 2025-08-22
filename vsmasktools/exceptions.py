from __future__ import annotations

from typing import Any, ClassVar

from jetpytools import fallback

from vstools import CustomValueError, FuncExcept


class _UnknownMaskDetectError(CustomValueError):
    _placeholder: ClassVar[str]
    _message: ClassVar[str]

    def __init__(self, func: FuncExcept, name: str, message: str | None = None, **kwargs: Any) -> None:
        """
        Instantiate a new exception with pretty printing and more.

        Args:
            func: Function this exception was raised from.
            name: EdgeDetect kind.
            message: Message of the error.
        """
        super().__init__(fallback(message, self._message), func, **{self._placeholder: name}, **kwargs)


class UnknownEdgeDetectError(_UnknownMaskDetectError):
    """
    Raised when an unknown edge detect is passed.
    """

    _placeholder = "edge_detect"
    _message = 'Unknown concrete edge detector "{edge_detect}"!'


class UnknownRidgeDetectError(_UnknownMaskDetectError):
    """
    Raised when an unknown ridge detect is passed.
    """

    _placeholder = "ridge_detect"
    _message = 'Unknown concrete ridge detector "{ridge_detect}"!'
