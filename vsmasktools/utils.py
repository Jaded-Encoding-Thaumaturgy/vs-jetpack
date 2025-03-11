from __future__ import annotations

from typing import Any, Callable, Concatenate, Iterable, overload

from vsexprtools import ExprOp, complexpr_available, norm_expr
from vskernels import Bilinear, Kernel, KernelT
from vsrgtools import box_blur, gauss_blur
from vstools import (
    ColorRange, CustomValueError, FrameRangeN, FrameRangesN, FuncExceptT, P, check_ref_clip,
    check_variable, check_variable_format, core, depth, flatten_vnodes, get_lowest_values, get_peak_values,
    insert_clip, normalize_ranges, plane, replace_ranges, split, vs
)

from .abstract import GeneralMask
from .edge import EdgeDetect, EdgeDetectT, RidgeDetect, RidgeDetectT
from .types import GenericMaskT

__all__ = [
    'max_planes',

    'region_rel_mask', 'region_abs_mask',

    'squaremask', 'replace_squaremask', 'freeze_replace_squaremask',

    'normalize_mask',

    'rekt_partial'
]


def max_planes(*_clips: vs.VideoNode | Iterable[vs.VideoNode], resizer: KernelT = Bilinear) -> vs.VideoNode:
    clips = flatten_vnodes(_clips)

    assert check_variable_format((model := clips[0]), max_planes)

    resizer = Kernel.ensure_obj(resizer, max_planes)

    width, height, fmt = model.width, model.height, model.format.replace(subsampling_w=0, subsampling_h=0)

    return ExprOp.MAX.combine(
        split(resizer.scale(clip, width, height, format=fmt)) for clip in clips
    )


def _get_region_expr(
    clip: vs.VideoNode | vs.VideoFrame, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0,
    replace: str | int = 0, rel: bool = False
) -> str:
    right, bottom = right + 1, bottom + 1

    if isinstance(replace, int):
        replace = f'{replace} x'

    if rel:
        return f'X {left} < X {right} > or Y {top} < Y {bottom} > or or {replace} ?'

    return f'X {left} < X {clip.width - right} > or Y {top} < Y {clip.height - bottom} > or or {replace} ?'


def region_rel_mask(clip: vs.VideoNode, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0) -> vs.VideoNode:
    if complexpr_available:
        return norm_expr(
            clip, _get_region_expr(clip, left, right, top, bottom, 0), func=region_rel_mask
        )

    return clip.std.Crop(left, right, top, bottom).std.AddBorders(left, right, top, bottom)


def region_abs_mask(clip: vs.VideoNode, width: int, height: int, left: int = 0, top: int = 0) -> vs.VideoNode:
    def _crop(w: int, h: int) -> vs.VideoNode:
        return clip.std.CropAbs(width, height, left, top).std.AddBorders(
            left, w - width - left, top, h - height - top
        )

    if 0 in {clip.width, clip.height}:
        if complexpr_available:
            return norm_expr(
                clip, _get_region_expr(clip, left, left + width, top, top + height, 0, True),
                func=region_rel_mask
            )

        return clip.std.FrameEval(lambda f, n: _crop(f.width, f.height), clip)

    return region_rel_mask(clip, left, clip.width - width - left, top, clip.height - height - top)


def squaremask(
    clip: vs.VideoNode, width: int, height: int, offset_x: int, offset_y: int, invert: bool = False,
    force_gray: bool = True,
    func: FuncExceptT | None = None
) -> vs.VideoNode:
    """
    Create a square used for simple masking.

    This is a fast and simple mask that's useful for very rough and simple masking.

    :param clip:        The clip to process.
    :param width:       The width of the square. This must be less than clip.width - offset_x.
    :param height:      The height of the square. This must be less than clip.height - offset_y.
    :param offset_x:    The location of the square, offset from the left side of the frame.
    :param offset_y:    The location of the square, offset from the top of the frame.
    :param invert:      Invert the mask. This means everything *but* the defined square will be masked.
                        Default: False.
    :param force_gray:  Whether to force using GRAY format or clip format.
    :param func:        Function returned for custom error handling.
                        This should only be set by VS package developers.
                        Default: :py:func:`squaremask`.

    :return:            A mask in the shape of a square.
    """
    func = func or squaremask

    assert check_variable(clip, func)

    mask_format = clip.format.replace(
        color_family=vs.GRAY, subsampling_w=0, subsampling_h=0
    ) if force_gray else clip.format

    if offset_x + width > clip.width or offset_y + height > clip.height:
        raise CustomValueError('mask exceeds clip size!', func)

    if complexpr_available:
        base_clip = clip.std.BlankClip(
            None, None, mask_format.id, 1,
            color=get_lowest_values(mask_format, ColorRange.FULL),
            keep=True
        )
        exprs = [
            _get_region_expr(
                base_clip, offset_x, clip.width - width - offset_x, offset_y, clip.height - height - offset_y,
                'range_max x' if invert else 'x range_max'
            )
        ]

        if mask_format.num_planes > 1:
            for i in range(1, mask_format.num_planes):
                p = plane(base_clip, i)
                ratio_x = p.width / base_clip.width
                ratio_y = p.height / base_clip.height
                exprs.append(
                        _get_region_expr(
                        p,
                        int(offset_x * ratio_x), int((clip.width - width - offset_x) * ratio_x),
                        int(offset_y * ratio_y), int((clip.height - height - offset_y) * ratio_y),
                        'range_max x' if invert else 'x range_max'
                    )
                )

        mask = norm_expr(base_clip, tuple(exprs), func=func)
    else:
        base_clip = clip.std.BlankClip(
            width, height, mask_format.id, 1, color=get_peak_values(mask_format, ColorRange.FULL), keep=True
        )

        mask = base_clip.std.AddBorders(
            offset_x, clip.width - width - offset_x, offset_y, clip.height - height - offset_y,
            [0] * mask_format.num_planes
        )
        if invert:
            mask = mask.std.Invert()

    if clip.num_frames == 1:
        return mask

    return mask.std.Loop(clip.num_frames)


def replace_squaremask(
    clipa: vs.VideoNode, clipb: vs.VideoNode, mask_params: tuple[int, int, int, int],
    ranges: FrameRangeN | FrameRangesN | None = None, blur_sigma: int | float | None = None,
    invert: bool = False, func: FuncExceptT | None = None, show_mask: bool = False
) -> vs.VideoNode:
    """
    Replace an area of the frame with another clip using a simple square mask.

    This is a convenience wrapper merging square masking and framerange replacing functionalities
    into one function, along with additional utilities such as blurring.

    :param clipa:           Base clip to process.
    :param clipb:           Clip to mask on top of `clipa`.
    :param mask_params:     Parameters passed to `squaremask`. Expects a tuple of (width, height, offset_x, offset_y).
    :param ranges:          Frameranges to replace with the masked clip. If `None`, replaces the entire clip.
                            Default: None.
    :param blur_sigma:      Post-blurring of the mask to help hide hard edges.
                            If you pass an int, a :py:func:`box_blur` will be used.
                            Passing a float will use a :py:func:`gauss_blur` instead.
                            Default: None.
    :param invert:          Invert the mask. This means everything *but* the defined square will be masked.
                            Default: False.
    :param func:            Function returned for custom error handling.
                            This should only be set by VS package developers.
                            Default: :py:func:`squaremask`.
    :param show_mask:       Return the mask instead of the masked clip.

    :return:                Clip with a squaremask applied, and optionally set to specific frameranges.
    """
    func = func or replace_squaremask

    assert check_variable(clipa, func) and check_variable(clipb, func)

    mask = squaremask(clipb[0], *mask_params, invert, func=func)

    if isinstance(blur_sigma, int):
        mask = box_blur(mask, blur_sigma)
    elif isinstance(blur_sigma, float):
        mask = gauss_blur(mask, blur_sigma)

    mask = mask.std.Loop(clipa.num_frames)

    if show_mask:
        return mask

    merge = clipa.std.MaskedMerge(clipb, mask)

    ranges = normalize_ranges(clipa, ranges)

    if len(ranges) == 1 and ranges[0] == (0, clipa.num_frames - 1):
        return merge

    return replace_ranges(clipa, merge, ranges)


def freeze_replace_squaremask(
    mask: vs.VideoNode, insert: vs.VideoNode, mask_params: tuple[int, int, int, int],
    frame: int, frame_range: tuple[int, int]
) -> vs.VideoNode:
    start, end = frame_range

    masked_insert = replace_squaremask(mask[frame], insert[frame], mask_params)

    return insert_clip(mask, masked_insert * (end - start + 1), start)


@overload
def normalize_mask(mask: vs.VideoNode, clip: vs.VideoNode) -> vs.VideoNode:
    ...


@overload
def normalize_mask(
    mask: Callable[[vs.VideoNode, vs.VideoNode], vs.VideoNode], clip: vs.VideoNode, ref: vs.VideoNode
) -> vs.VideoNode:
    ...


@overload
def normalize_mask(
    mask: EdgeDetectT | RidgeDetectT, clip: vs.VideoNode, *, ridge: bool = ..., **kwargs: Any
) -> vs.VideoNode:
    ...


@overload
def normalize_mask(mask: GeneralMask, clip: vs.VideoNode, ref: vs.VideoNode) -> vs.VideoNode:
    ...


@overload
def normalize_mask(
    mask: GenericMaskT, clip: vs.VideoNode, ref: vs.VideoNode | None = ..., *, ridge: bool = ..., **kwargs: Any
) -> vs.VideoNode:
    ...


def normalize_mask(
    mask: GenericMaskT, clip: vs.VideoNode, ref: vs.VideoNode | None = None,
    *, ridge: bool = False, **kwargs: Any
) -> vs.VideoNode:
    if isinstance(mask, (str, type)):
        return normalize_mask(EdgeDetect.ensure_obj(mask, normalize_mask), clip, ref, ridge=ridge, **kwargs)

    if isinstance(mask, EdgeDetect):
        if ridge and isinstance(mask, RidgeDetect):
            cmask = mask.ridgemask(clip, **kwargs)
        else:
            cmask = mask.edgemask(clip, **kwargs)
    elif isinstance(mask, GeneralMask):
        cmask = mask.get_mask(clip, ref)
    elif callable(mask):
        if ref is None:
            raise CustomValueError('This mask function requires a ref to be specified!')

        cmask = mask(clip, ref)
    else:
        cmask = mask

    return depth(cmask, clip, range_in=ColorRange.FULL, range_out=ColorRange.FULL)


def rekt_partial(
    clip: vs.VideoNode, left: int = 0, top: int = 0, right: int = 0, bottom: int = 0,
    func: Callable[Concatenate[vs.VideoNode, P], vs.VideoNode] = lambda clip, *args, **kwargs: clip,
    *args: P.args, **kwargs: P.kwargs
) -> vs.VideoNode:
    assert check_variable(clip, rekt_partial)

    if left == top == right == bottom == 0:
        return func(clip, *args, **kwargs)

    cropped = clip.std.Crop(left, right, top, bottom)

    filtered = func(cropped, *args, **kwargs)

    check_ref_clip(cropped, filtered, rekt_partial)

    if complexpr_available:
        filtered = filtered.std.AddBorders(left, right, top, bottom)

        ratio_w, ratio_h = 1 << clip.format.subsampling_w, 1 << clip.format.subsampling_h

        vals = list(filter(None, [
            ('X {left} >= ' if left else None),
            ('X {right} < ' if right else None),
            ('Y {top} >= ' if top else None),
            ('Y {bottom} < ' if bottom else None)
        ]))

        return norm_expr(
            [clip, filtered], [*vals, ['and'] * (len(vals) - 1), 'y x ?'],
            left=[left, left / ratio_w], right=[clip.width - right, (clip.width - right) / ratio_w],
            top=[top, top / ratio_h], bottom=[clip.height - bottom, (clip.height - bottom) / ratio_h],
            func=rekt_partial
        )

    if not (top or bottom) and (right or left):
        return core.std.StackHorizontal(list(filter(None, [
            clip.std.CropAbs(left, clip.height) if left else None,
            filtered,
            clip.std.CropAbs(right, clip.height, x=clip.width - right) if right else None,
        ])))

    if (top or bottom) and (right or left):
        filtered = core.std.StackHorizontal(list(filter(None, [
            clip.std.CropAbs(left, filtered.height, y=top) if left else None,
            filtered,
            clip.std.CropAbs(right, filtered.height, x=clip.width - right, y=top) if right else None,
        ])))

    return core.std.StackVertical(list(filter(None, [
        clip.std.CropAbs(clip.width, top) if top else None,
        filtered,
        clip.std.CropAbs(clip.width, bottom, y=clip.height - bottom) if bottom else None,
    ])))
