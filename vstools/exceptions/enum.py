from __future__ import annotations

from jetpytools import CustomValueError, NotFoundEnumValue

__all__ = [
    "NotFoundEnumValue",
    "UndefinedChromaLocationError",
    "UndefinedFieldBasedError",
    "UnsupportedChromaLocationError",
    "UnsupportedFieldBasedError",
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
