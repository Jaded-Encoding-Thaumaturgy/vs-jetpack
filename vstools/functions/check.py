from __future__ import annotations

import inspect

from functools import partial, wraps
from typing import Any, Callable, Literal, Sequence, TypeGuard, cast, overload

import vapoursynth as vs

from jetpytools import CustomError, F, FuncExceptT

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

    'disallow_variable_format',
    'disallow_variable_resolution',
]


def _check_variable(
    function: F, vname: str, error: type[CustomError],
    only_first: bool, check_func: Callable[[vs.VideoNode], bool]
) -> F:
    def _check(x: Any) -> bool:
        return isinstance(x, vs.VideoNode) and check_func(x)

    @wraps(function)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        if only_first:
            import warnings
            warnings.warn("Please use the check functions if only_first is True", SyntaxWarning)

        for obj in args[:1] if only_first else [*args, *kwargs.values()]:
            if _check(obj):
                raise error(func=function)

        if not only_first:
            for name, param in inspect.signature(function).parameters.items():
                if param.default is not inspect.Parameter.empty and _check(param.default):
                    raise error(
                        message=f'Variable-{vname} clip not allowed in default argument `{name}`.', func=function
                    )

        return function(*args, **kwargs)

    return cast(F, _wrapper)


@overload
def disallow_variable_format(*, only_first: Literal[False] = ...) -> Callable[[F], F]:
    ...


@overload
def disallow_variable_format(function: F | None = None, /) -> F:
    ...


def disallow_variable_format(function: F | None = None, /, *, only_first: Literal[False] = False) -> Callable[[F], F] | F:
    """
    Decorator for disallowing clips with variable formats.

    :param only_first:              Only verify the format of the first argument.
                                    Default: False.

    :raises VariableFormatError:    A clip with a variable format is found.
    """

    if function is None:
        return cast(Callable[[F], F], partial(disallow_variable_format, only_first=only_first))

    return _check_variable(
        function, 'format', VariableFormatError, only_first, lambda x: x.format is None
    )


@overload
def disallow_variable_resolution(*, only_first: Literal[False] = ...) -> Callable[[F], F]:
    ...


@overload
def disallow_variable_resolution(func: F | None = None, /) -> F:
    ...


def disallow_variable_resolution(function: F | None = None, /, *, only_first: Literal[False] = False) -> Callable[[F], F] | F:
    """
    Decorator for disallowing clips with variable resolutions.

    :param only_first:                  Only verify the resolution of the first argument.
                                        Default: False.

    :raises VariableResolutionError:    A clip with a variable resolution is found.
    """

    if function is None:
        return cast(Callable[[F], F], partial(disallow_variable_resolution, only_first=only_first))

    return _check_variable(
        function, 'resolution', VariableResolutionError, only_first, lambda x: not all({x.width, x.height})
    )


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
