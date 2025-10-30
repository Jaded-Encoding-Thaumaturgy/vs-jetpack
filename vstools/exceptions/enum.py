from __future__ import annotations

from jetpytools import CustomValueError

__all__ = [
    "UndefinedChromaLocationError",
    "UndefinedFieldBasedError",
    "UndefinedFieldError",
    "UnsupportedChromaLocationError",
    "UnsupportedFieldBasedError",
    "UnsupportedFieldError",
]


class UndefinedChromaLocationError(CustomValueError):
    """
    Raised when an undefined chroma location is passed.
    """


class UnsupportedChromaLocationError(CustomValueError):
    """
    Raised when an unsupported chroma location is passed.
    """


class UndefinedFieldBasedError(CustomValueError):
    """
    Raised when an undefined field type is passed.
    """


class UnsupportedFieldBasedError(CustomValueError):
    """
    Raised when an unsupported field type is passed.
    """


class UndefinedFieldError(CustomValueError):
    """
    Raised when an undefined field is passed.
    """


class UnsupportedFieldError(CustomValueError):
    """
    Raised when an unsupported field is passed.
    """
