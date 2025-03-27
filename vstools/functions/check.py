from __future__ import annotations

from typing import Sequence, TypeGuard, overload

import vapoursynth as vs

from jetpytools import FuncExceptT

from ..enums import FieldBased
from ..exceptions import (
    FormatsRefClipMismatchError, ResolutionsRefClipMismatchError, UnsupportedFieldBasedError,
    VariableFormatError, VariableResolutionError
)
from ..types import ConstantFormatVideoNode, VideoNodeT

__all__ = [
    'check_ref_clip',

    'check_variable',
    'check_variable_format',
    'check_variable_resolution',
    'check_correct_subsampling',
    'check_progressive',
]


def check_ref_clip(src: vs.VideoNode, ref: vs.VideoNode | None, func: FuncExceptT | None = None) -> vs.VideoNode:
    """
    Function for ensuring the ref clip's format matches that of the input clip.

    If no ref clip can be found, this function will simply do nothing.

    :param src:     Input clip.
    :param ref:     Reference clip.
                    Default: None.

    :raises VariableFormatError                 The format of either clip is variable.
    :raises VariableResolutionError:            The resolution of either clip is variable.
    :raises FormatsRefClipMismatchError:        The formats of the two clips do not match.
    :raises ResolutionsRefClipMismatchError:    The resolutions of the two clips do not match.

    :return:        Ref clip.
    """

    if ref is None:
        return src

    func = func or check_ref_clip

    assert check_variable(src, func)
    assert check_variable(ref, func)

    FormatsRefClipMismatchError.check(func, src, ref)
    ResolutionsRefClipMismatchError.check(func, src, ref)

    return ref


@overload
def check_variable_format(clip: vs.VideoNode, func: FuncExceptT) -> TypeGuard[ConstantFormatVideoNode]:
    ...


@overload
def check_variable_format(clip: Sequence[vs.VideoNode], func: FuncExceptT) -> TypeGuard[Sequence[ConstantFormatVideoNode]]:
    ...


def check_variable_format(
    clip: vs.VideoNode | Sequence[vs.VideoNode], func: FuncExceptT
) -> TypeGuard[ConstantFormatVideoNode] | TypeGuard[Sequence[ConstantFormatVideoNode]]:
    """
    Check for variable format and return an error if found.

    :raises VariableFormatError:    The clip is of a variable format.
    """
    clip = [clip] if isinstance(clip, vs.VideoNode) else clip

    for c in clip:
        if c.format is None:
            raise VariableFormatError(func)

    return True


def check_variable_resolution(clip: VideoNodeT, func: FuncExceptT) -> TypeGuard[VideoNodeT]:
    """
    Check for variable width or height and return an error if found.

    :raises VariableResolutionError:    The clip has a variable resolution.
    """

    if 0 in (clip.width, clip.height):
        raise VariableResolutionError(func)

    return True


def check_variable(clip: vs.VideoNode, func: FuncExceptT) -> TypeGuard[ConstantFormatVideoNode]:
    """
    Check for variable format and a variable resolution and return an error if found.

    :raises VariableFormatError:        The clip is of a variable format.
    :raises VariableResolutionError:    The clip has a variable resolution.
    """

    check_variable_format(clip, func)
    check_variable_resolution(clip, func)

    return True


def check_correct_subsampling(
    clip: vs.VideoNode, width: int | None = None, height: int | None = None, func: FuncExceptT | None = None
) -> None:
    """
    Check if the subsampling is correct and return an error if it's not.

    :param clip:                        Clip to check.
    :param width:                       Output width.
    :param height:                      Output height.
    :param func:                        Function returned for custom error handling.
                                        This should only be set by VS package developers.

    :raises InvalidSubsamplingError:    The clip has invalid subsampling.
    """
    from ..exceptions import InvalidSubsamplingError

    if clip.format:
        if (
            (width is not None and width % (1 << clip.format.subsampling_w))
            or (height is not None and height % (1 << clip.format.subsampling_h))
        ):
            raise InvalidSubsamplingError(
                func or check_correct_subsampling, clip,
                'The {subsampling} subsampling is not supported for this resolution!',
                reason=dict(width=width, height=height)
            )

def check_progressive(clip: VideoNodeT, func: FuncExceptT) -> TypeGuard[VideoNodeT]:
    """
    Check if the clip is progressive and return an error if it's not.

    :param clip:                        Clip to check.
    :param func:                        Function returned for custom error handling.
                                        This should only be set by VS package developers.

    :raises UnsupportedFieldBasedError: The clip is interlaced.
    """

    if FieldBased.from_video(clip, func=func).is_inter:
        raise UnsupportedFieldBasedError("Only progressive video is supported!", func)

    return True
