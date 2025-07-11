from __future__ import annotations

from typing import Any, ClassVar

from jetpytools import fallback

from vstools import CustomValueError, FuncExceptT

__all__ = ["UnknownDescalerError", "UnknownKernelError", "UnknownResamplerError", "UnknownScalerError"]


class _UnknownBaseScalerError(CustomValueError):
    """
    Base Scaler error class
    """

    _placeholder: ClassVar[str]
    _message: ClassVar[str]

    def __init__(self, func: FuncExceptT, name: str, message: str | None = None, **kwargs: Any) -> None:
        """
        Instantiate a new exception with pretty printing and more.

        Args:
            func: Function this exception was raised from.
            name: Base scaler name.
            message: Message of the error.
        """
        super().__init__(fallback(message, self._message), func, **{self._placeholder: name}, **kwargs)


class UnknownScalerError(_UnknownBaseScalerError):
    """
    Raised when an unknown scaler is passed.
    """

    _placeholder = "scaler"
    _message = 'Unknown concrete scaler "{scaler}"!'


class UnknownDescalerError(_UnknownBaseScalerError):
    """
    Raised when an unknown descaler is passed.
    """

    _placeholder = "descaler"
    _message = 'Unknown concrete descaler "{descaler}"!'


class UnknownResamplerError(_UnknownBaseScalerError):
    """
    Raised when an unknown resampler is passed.
    """

    _placeholder = "resampler"
    _message = 'Unknown concrete resampler "{resampler}"!'


class UnknownKernelError(_UnknownBaseScalerError):
    """
    Raised when an unknown kernel is passed.
    """

    _placeholder = "kernel"
    _message = 'Unknown concrete kernel "{kernel}"!'
