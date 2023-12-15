from __future__ import annotations

import inspect
from functools import partial, wraps
from typing import Any, Callable, cast, overload

from stgpytools import CustomValueError, FuncExceptT, fallback

from ..enums import (
    ChromaLocation, ChromaLocationT, ColorRange, ColorRangeT, FieldBased, FieldBasedT, Matrix, MatrixT, Primaries,
    PrimariesT, PropEnum, Transfer, TransferT
)
from ..functions import DitherType, check_variable, depth
from ..types import F_VD, HoldsVideoFormatT, VideoFormatT
from . import vs_proxy as vs
from .scale import scale_8bit

__all__ = [
    'finalize_clip',
    'finalize_output',
    'initialize_clip',
    'initialize_input'
]


def finalize_clip(
    clip: vs.VideoNode,
    bits: VideoFormatT | HoldsVideoFormatT | int | None = 10,
    clamp_tv_range: bool | None = None,
    dither_type: DitherType = DitherType.AUTO,
    *, func: FuncExceptT | None = None
) -> vs.VideoNode:
    """
    Finalize a clip for output to the encoder.

    :param clip:            Clip to output.
    :param bits:            Bitdepth to output to.
    :param clamp_tv_range:  Whether to clamp to tv range. If None, decide based on clip properties.
    :param dither_type:     Dithering used for the bitdepth conversion.
    :param func:            Optional function this was called from.

    :return:                Dithered down and optionally clamped clip.
    """

    assert check_variable(clip, func or finalize_clip)

    if bits:
        clip = depth(clip, bits, dither_type=dither_type)

    if clamp_tv_range is None:
        try:
            clamp_tv_range = ColorRange.from_video(clip, strict=True).is_limited
        except Exception:
            clamp_tv_range = True

    if clamp_tv_range:
        low_luma, high_luma = scale_8bit(clip, 16), scale_8bit(clip, 235)
        low_chroma, high_chroma = scale_8bit(clip, 16, True), scale_8bit(clip, 240, True)

        if hasattr(vs.core, 'akarin'):
            clip = clip.akarin.Expr([
                f'x {low_luma} {high_luma} clamp', f'x {low_chroma} {high_chroma} clamp'
            ])
        else:
            clip = clip.std.Expr([
                f'x {low_luma} max {high_luma} min', f'x {low_chroma} max {high_chroma} min'
            ])

    return clip


@overload
def finalize_output(
    function: None = None, /, *, bits: int | None = 10,
    clamp_tv_range: bool = True, dither_type: DitherType = DitherType.AUTO, func: FuncExceptT | None = None
) -> Callable[[F_VD], F_VD] | F_VD:
    ...


@overload
def finalize_output(
    function: F_VD, /, *, bits: int | None = 10,
    clamp_tv_range: bool = True, dither_type: DitherType = DitherType.AUTO, func: FuncExceptT | None = None
) -> F_VD:
    ...


def finalize_output(
    function: F_VD | None = None, /, *, bits: int | None = 10,
    clamp_tv_range: bool = True, dither_type: DitherType = DitherType.AUTO, func: FuncExceptT | None = None
) -> Callable[[F_VD], F_VD] | F_VD:
    """Decorator implementation of finalize_clip."""

    if function is None:
        return cast(
            Callable[[F_VD], F_VD],
            partial(finalize_output, bits=bits, clamp_tv_range=clamp_tv_range, dither_type=dither_type, func=func)
        )

    @wraps(function)
    def _wrapper(*args: Any, **kwargs: Any) -> vs.VideoNode:
        assert function
        return finalize_clip(function(*args, **kwargs), bits, clamp_tv_range, dither_type, func=func)

    return cast(F_VD, _wrapper)


def initialize_clip(
    clip: vs.VideoNode, bits: int | None = 16,
    matrix: MatrixT | None = None,
    transfer: TransferT | None = None,
    primaries: PrimariesT | None = None,
    chroma_location: ChromaLocationT | None = None,
    color_range: ColorRangeT | None = None,
    field_based: FieldBasedT | None = None,
    strict: bool = False,
    dither_type: DitherType = DitherType.AUTO, *, func: FuncExceptT | None = None
) -> vs.VideoNode:
    """
    Initialize a clip with default props.

    :param clip:            Clip to initialize.
    :param bits:            Bits to dither to. If None, no dithering is applied.
    :param matrix:          Matrix property to set. If None, tries to get the Matrix from existing props.
                            If no props are set or Matrix=2, guess from the video resolution.
    :param transfer:        Transfer property to set. If None, tries to get the Transfer from existing props.
                            If no props are set or Transfer=2, guess from the video resolution.
    :param primaries:       Primaries property to set. If None, tries to get the Primaries from existing props.
                            If no props are set or Primaries=2, guess from the video resolution.
    :param chroma_location: ChromaLocation prop to set. If None, tries to get the ChromaLocation from existing props.
                            If no props are set, guess from the video resolution.
    :param color_range:     ColorRange prop to set. If None, tries to get the ColorRange from existing props.
                            If no props are set, assume Limited Range.
    :param field_based:     FieldBased prop to set. If None, tries to get the FieldBased from existing props.
    :param strict:          Whether to be strict about existing properties.
                            If True, throws an exception if certain frame properties are not found.
    :param dither_type:     Dithering used for the bitdepth conversion.
    :param func:            Optional function this was called from.

    :return:                Clip with relevant frame properties set, and optionally dithered up to 16 bits.
    """

    func = fallback(func, initialize_clip)  # type: ignore

    values: list[tuple[type[PropEnum], Any]] = [
        (Matrix, matrix),
        (Transfer, transfer),
        (Primaries, primaries),
        (ChromaLocation, chroma_location),
        (ColorRange, color_range),
        (FieldBased, field_based)
    ]

    clip = PropEnum.ensure_presences(clip, [
        (cls if strict else cls.from_video(clip, False, func)) if value is None else cls.from_param(value, func)
        for cls, value in values
    ], func)

    if bits is None:
        return clip

    return depth(clip, bits, dither_type=dither_type)


@overload
def initialize_input(
    function: None = None, /, *, bits: int | None = 16,
    matrix: MatrixT | None = None,
    transfer: TransferT | None = None,
    primaries: PrimariesT | None = None,
    chroma_location: ChromaLocationT | None = None,
    color_range: ColorRangeT | None = None,
    field_based: FieldBasedT | None = None,
    dither_type: DitherType = DitherType.AUTO,
    func: FuncExceptT | None = None
) -> Callable[[F_VD], F_VD]:
    ...


@overload
def initialize_input(
    function: F_VD, /, *, bits: int | None = 16,
    matrix: MatrixT | None = None,
    transfer: TransferT | None = None,
    primaries: PrimariesT | None = None,
    chroma_location: ChromaLocationT | None = None,
    color_range: ColorRangeT | None = None,
    field_based: FieldBasedT | None = None,
    strict: bool = False,
    dither_type: DitherType = DitherType.AUTO, func: FuncExceptT | None = None
) -> F_VD:
    ...


def initialize_input(
    function: F_VD | None = None, /, *, bits: int | None = 16,
    matrix: MatrixT | None = None,
    transfer: TransferT | None = None,
    primaries: PrimariesT | None = None,
    chroma_location: ChromaLocationT | None = None,
    color_range: ColorRangeT | None = None,
    field_based: FieldBasedT | None = None,
    strict: bool = False,
    dither_type: DitherType = DitherType.AUTO, func: FuncExceptT | None = None
) -> Callable[[F_VD], F_VD] | F_VD:
    """
    Decorator implementation of ``initialize_clip``
    """

    init_args = dict[str, Any](
        bits=bits,
        matrix=matrix, transfer=transfer, primaries=primaries,
        chroma_location=chroma_location, color_range=color_range,
        field_based=field_based, strict=strict, dither_type=dither_type, func=func
    )

    if function is None:
        return cast(Callable[[F_VD], F_VD], partial(initialize_input, **init_args))

    @wraps(function)
    def _wrapper(*args: Any, **kwargs: Any) -> vs.VideoNode:
        assert function

        args_l = list(args)
        for i, obj in enumerate(args_l):
            if isinstance(obj, vs.VideoNode):
                args_l[i] = initialize_clip(obj, **init_args)
                return function(*args_l, **kwargs)

        kwargs2 = kwargs.copy()
        for name, obj in kwargs2.items():
            if isinstance(obj, vs.VideoNode):
                kwargs2[name] = initialize_clip(obj, **init_args)
                return function(*args, **kwargs2)

        for name, param in inspect.signature(function).parameters.items():
            if param.default is not inspect.Parameter.empty and isinstance(param.default, vs.VideoNode):
                return function(*args, **kwargs2 | {name: initialize_clip(param.default, **init_args)})

        raise CustomValueError(
            'No VideoNode found in positional, keyword nor default arguments!', func or initialize_input
        )

    return cast(F_VD, _wrapper)
