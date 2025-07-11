from __future__ import annotations

from typing import Any

import vapoursynth as vs
from jetpytools import CustomPermissionError, CustomValueError, FuncExceptT, SupportsString

__all__ = [
    "InvalidColorspacePathError",
    "InvalidMatrixError",
    "InvalidPrimariesError",
    "InvalidTransferError",
    "ReservedMatrixError",
    "ReservedPrimariesError",
    "ReservedTransferError",
    "UndefinedMatrixError",
    "UndefinedPrimariesError",
    "UndefinedTransferError",
    "UnsupportedColorRangeError",
    "UnsupportedMatrixError",
    "UnsupportedPrimariesError",
    "UnsupportedTransferError",
]

########################################################
# Colorspace


class InvalidColorspacePathError(CustomValueError):
    """
    Raised when there is no path between two colorspaces.
    """

    def __init__(self, func: FuncExceptT, message: SupportsString | None = None, **kwargs: Any) -> None:
        def_msg = "Unable to convert between colorspaces! "
        def_msg += "Please provide more colorspace information (e.g., matrix, transfer, primaries)."

        if isinstance(message, vs.Error):
            error_msg = str(message)
            if "Resize error:" in error_msg:
                kwargs["reason"] = error_msg[error_msg.find("(") + 1 : error_msg.rfind(")")]
                message = def_msg

        super().__init__(message or def_msg, func, **kwargs)

    @staticmethod
    def check(func: FuncExceptT, to_check: vs.VideoNode) -> None:
        """
        Check if there's a valid colorspace path for the given clip.

        Args:
            func: Function returned for custom error handling. This should only be set by VS package developers.
            to_check: Value to check. Must be a VideoNode.

        Raises:
            InvalidColorspacePathError: If there's no valid colorspace path.
        """

        try:
            to_check.get_frame(0).close()
        except vs.Error as e:
            if "no path between colorspaces" in str(e):
                raise InvalidColorspacePathError(func, e)
            raise


########################################################
# Matrix


class UndefinedMatrixError(CustomValueError):
    """
    Raised when an undefined matrix is passed.
    """


class ReservedMatrixError(CustomPermissionError):
    """
    Raised when a reserved matrix is requested.
    """


class UnsupportedMatrixError(CustomValueError):
    """
    Raised when an unsupported matrix is passed.
    """


class InvalidMatrixError(CustomValueError):
    """
    Raised when an invalid matrix is passed.
    """

    def __init__(
        self,
        func: FuncExceptT,
        matrix: int = 2,
        message: SupportsString = "You can't set a matrix of {matrix}!",
        **kwargs: Any,
    ) -> None:
        super().__init__(message, func, matrix=matrix, **kwargs)


########################################################
# Transfer


class UndefinedTransferError(CustomValueError):
    """
    Raised when an undefined transfer is passed.
    """


class ReservedTransferError(CustomPermissionError):
    """
    Raised when a reserved transfer is requested.
    """


class UnsupportedTransferError(CustomValueError):
    """
    Raised when an unsupported transfer is passed.
    """


class InvalidTransferError(CustomValueError):
    """
    Raised when an invalid matrix is passed.
    """

    def __init__(
        self,
        func: FuncExceptT,
        transfer: int = 2,
        message: SupportsString = "You can't set a transfer of {transfer}!",
        **kwargs: Any,
    ) -> None:
        super().__init__(message, func, transfer=transfer, **kwargs)


########################################################
# Primaries


class UndefinedPrimariesError(CustomValueError):
    """
    Raised when an undefined primaries value is passed.
    """


class ReservedPrimariesError(CustomPermissionError):
    """
    Raised when reserved primaries are requested.
    """


class UnsupportedPrimariesError(CustomValueError):
    """
    Raised when a unsupported primaries value is passed.
    """


class InvalidPrimariesError(CustomValueError):
    """
    Raised when an invalid matrix is passed.
    """

    def __init__(
        self,
        func: FuncExceptT,
        primaries: int = 2,
        message: SupportsString = "You can't set primaries of {primaries}!",
        **kwargs: Any,
    ) -> None:
        super().__init__(message, func, primaries=primaries, **kwargs)


########################################################
# ColorRange


class UnsupportedColorRangeError(CustomValueError):
    """
    Raised when a unsupported color range value is passed.
    """
