from __future__ import annotations

from typing import Any

from vstools import CustomValueError, FuncExceptT

__all__ = [
    "UnknownEdgeDetectError",
    "UnknownRidgeDetectError",
]


class UnknownEdgeDetectError(CustomValueError):
    """
    Raised when an unknown edge detect is passed.
    """

    def __init__(
        self,
        func: FuncExceptT,
        edge_detect: str,
        message: str = 'Unknown concrete edge detector "{edge_detect}"!',
        **kwargs: Any,
    ) -> None:
        super().__init__(message, func, edge_detect=edge_detect, **kwargs)


class UnknownRidgeDetectError(CustomValueError):
    """
    Raised when an unknown ridge detect is passed.
    """

    def __init__(
        self,
        func: FuncExceptT,
        ridge_detect: str,
        message: str = 'Unknown concrete ridge detector "{ridge_detect}"!',
        **kwargs: Any,
    ) -> None:
        super().__init__(message, func, ridge_detect=ridge_detect, **kwargs)
