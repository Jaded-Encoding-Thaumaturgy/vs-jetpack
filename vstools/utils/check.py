from __future__ import annotations

from typing import Literal, Sequence

from jetpytools import CustomValueError, FuncExcept
from typing_extensions import deprecated

from ..enums import FieldBased
from ..exceptions import (
    FormatsRefClipMismatchError,
    ResolutionsRefClipMismatchError,
    UnsupportedFieldBasedError,
    VariableFormatError,
    VariableResolutionError,
)
from ..vs_proxy import vs

__all__ = [
    "check_correct_subsampling",
    "check_progressive",
    "check_ref_clip",
    "check_variable",
    "check_variable_format",
    "check_variable_resolution",
]


def check_ref_clip(src: vs.VideoNode, ref: vs.VideoNode | None, func: FuncExcept | None = None) -> vs.VideoNode:
    """
    Function for ensuring the ref clip's format matches that of the input clip.

    If no ref clip can be found, this function will simply do nothing.

    Args:
        src: Input clip.
        ref: Reference clip. Default: None.

    Raises:
        VariableFormatError: The format of either clip is variable.
        VariableResolutionError: The resolution of either clip is variable.
        FormatsRefClipMismatchError: The formats of the two clips do not match.
        ResolutionsRefClipMismatchError: The resolutions of the two clips do not match.

    Returns:
        Ref clip.
    """
    func = func or check_ref_clip

    assert check_variable(src, func)

    if ref is None:
        return src

    assert check_variable(ref, func)

    FormatsRefClipMismatchError.check(func, src, ref)
    ResolutionsRefClipMismatchError.check(func, src, ref)

    return ref


def check_variable_format(clip: vs.VideoNode | Sequence[vs.VideoNode], func: FuncExcept) -> Literal[True]:
    """
    Check for variable format and return an error if found.

    Raises:
        VariableFormatError: The clip is of a variable format.
    """
    clip = [clip] if isinstance(clip, vs.VideoNode) else clip

    for c in clip:
        if not c.format:
            raise VariableFormatError(func)

    return True


def check_variable_resolution(clip: vs.VideoNode, func: FuncExcept) -> Literal[True]:
    """
    Check for variable width or height and return an error if found.

    Raises:
        VariableResolutionError: The clip has a variable resolution.
    """

    if 0 in (clip.width, clip.height):
        raise VariableResolutionError(func)

    return True


def check_variable(clip: vs.VideoNode, func: FuncExcept) -> Literal[True]:
    """
    Check for variable format and a variable resolution and return an error if found.

    Raises:
        VariableFormatError: The clip is of a variable format.
        VariableResolutionError: The clip has a variable resolution.
    """

    check_variable_format(clip, func)
    check_variable_resolution(clip, func)

    return True


@deprecated(
    "check_correct_subsampling is deprecated and will be removed in a future version.", category=DeprecationWarning
)
def check_correct_subsampling(clip: vs.VideoNode, width: int, height: int, func: FuncExcept | None = None) -> None:
    """
    Check if the subsampling is correct and return an error if it's not.

    Args:
        clip: Clip to check.
        width: Output width.
        height: Output height.
        func: Function returned for custom error handling. This should only be set by VS package developers.

    Raises:
        CustomValueError: The clip has invalid subsampling.
    """
    if (width % (2**clip.format.subsampling_w)) or (height % (2**clip.format.subsampling_h)):
        from .info import get_subsampling

        raise CustomValueError(
            "The {subsampling} subsampling is not supported for this resolution!",
            func or check_correct_subsampling,
            {"width": width, "height": height},
            subsampling=get_subsampling(clip),
        )


def check_progressive(clip: vs.VideoNode, func: FuncExcept) -> Literal[True]:
    """
    Check if the clip is progressive and return an error if it's not.

    Args:
        clip: Clip to check.
        func: Function returned for custom error handling. This should only be set by VS package developers.

    Raises:
        UnsupportedFieldBasedError: The clip is interlaced.
    """

    if FieldBased.from_video(clip, func=func).is_inter:
        raise UnsupportedFieldBasedError("Only progressive video is supported!", func)

    return True
