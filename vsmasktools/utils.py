from __future__ import annotations

from typing import Any, Callable, Concatenate, Iterable, overload

from jetpytools import CustomValueError, FuncExcept, SupportsString, cachedproperty

from vsexprtools import ExprOp, norm_expr
from vskernels import Bilinear, Kernel, KernelLike
from vsrgtools import box_blur, gauss_blur
from vstools import (
    ColorRange,
    FrameRangeN,
    FrameRangesN,
    Planes,
    check_ref_clip,
    core,
    depth,
    flatten_vnodes,
    get_lowest_values,
    get_plane_sizes,
    insert_clip,
    normalize_ranges,
    replace_ranges,
    split,
    vs,
)

from .abstract import GeneralMask
from .edge import EdgeDetect, EdgeDetectLike, RidgeDetect, RidgeDetectLike
from .types import MaskLike

__all__ = [
    "freeze_replace_squaremask",
    "max_planes",
    "normalize_mask",
    "region_abs_mask",
    "region_rel_mask",
    "rekt_partial",
    "replace_squaremask",
    "squaremask",
]


def max_planes(*_clips: vs.VideoNode | Iterable[vs.VideoNode], resizer: KernelLike = Bilinear) -> vs.VideoNode:
    clips = flatten_vnodes(_clips)

    resizer = Kernel.ensure_obj(resizer, max_planes)

    width, height, fmt = clips[0].width, clips[0].height, clips[0].format.replace(subsampling_w=0, subsampling_h=0)

    return ExprOp.MAX.combine(split(resizer.scale(clip, width, height, format=fmt)) for clip in clips)


class RegionMask[**P, R]:
    """
    Class decorator that wraps [region_rel_mask][vsmasktools.region_rel_mask]
    and [region_abs_mask][vsmasktools.region_abs_mask] function and extends their functionality.

    It is not meant to be used directly.
    """

    def __init__(self, func: Callable[P, R]) -> None:
        self._func = func

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self._func(*args, **kwargs)

    @cachedproperty
    def expr(self) -> str:
        """
        Get the internal expr used for regioning.

        Returns:
            Expression.
        """
        return "X {left} < X {right} > or Y {top} < Y {bottom} > or or {replace_out} {replace_in} ?"


@RegionMask
def region_rel_mask(
    clip: vs.VideoNode,
    left: int = 0,
    right: int = 0,
    top: int = 0,
    bottom: int = 0,
    replace_in: SupportsString | None = None,
    replace_out: SupportsString | None = None,
    planes: Planes = None,
    func: FuncExcept | None = None,
) -> vs.VideoNode:
    """
    Generates a mask that defines a rectangular region within the clip, replacing pixels inside or outside the region,
    using relative coordinates.

    Args:
        clip: Input clip.
        left: Specifies how many pixels to crop from the left side. Defaults to 0.
        right: Specifies how many pixels to crop from the right side. Defaults to 0.
        top: Specifies how many pixels to crop from the top side. Defaults to 0.
        bottom: Specifies how many pixels to crop from the bottom side. Defaults to 0.
        replace_in: Pixel value or expression to fill inside the region.
            Defaults to using the original pixel values.
        replace_out: Pixel value or expression to fill outside the region.
            Defaults to the lowest possible values per plane.
        planes: Which planes to process. Defaults to all.
        func: Function returned for custom error handling. This should only be set by VS package developers.

    Returns:
        A new clip with the specified rectangular region masked in or out.
    """
    func = func or region_rel_mask

    if replace_in is None:
        replace_in = "x"
    if replace_out is None:
        replace_out = get_lowest_values(clip, ColorRange.FULL)

    lefts, rights, tops, bottoms = list[int](), list[int](), list[int](), list[int]()

    for i in range(clip.format.num_planes):
        w, h = get_plane_sizes(clip, i)
        scale_w = clip.width / w
        scale_h = clip.height / h

        lefts.append(int(left / scale_w))
        rights.append(int(w - (right / scale_w) - 1))
        tops.append(int(top / scale_h))
        bottoms.append(int(h - (bottom / scale_h) - 1))

    return norm_expr(
        clip,
        region_rel_mask.expr,
        planes,
        func=func,
        left=lefts,
        right=rights,
        top=tops,
        bottom=bottoms,
        replace_in=replace_in,
        replace_out=replace_out,
    )


@RegionMask
def region_abs_mask(
    clip: vs.VideoNode,
    width: int,
    height: int,
    left: int = 0,
    top: int = 0,
    replace_in: SupportsString | None = None,
    replace_out: SupportsString | None = None,
    planes: Planes = None,
    func: FuncExcept | None = None,
) -> vs.VideoNode:
    """
    Generates a mask that defines a rectangular region within the clip, replacing pixels inside or outside the region,
    using absolute coordinates.

    Args:
        clip: Input clip.
        width: The width of the region.
        height: The height of the region.
        left: Specifies how many pixels to crop from the left side. Defaults to 0.
        top: Specifies how many pixels to crop from the top side. Defaults to 0.
        replace_in: Pixel value or expression to fill inside the region.
            Defaults to using the original pixel values.
        replace_out: Pixel value or expression to fill outside the region.
            Defaults to the lowest possible values per plane.
        planes: Which planes to process. Defaults to all.
        func: Function returned for custom error handling. This should only be set by VS package developers.

    Returns:
        A new clip with the specified rectangular region masked in or out.
    """
    return region_rel_mask(
        clip,
        left,
        clip.width - width - left,
        top,
        clip.height - height - top,
        replace_in,
        replace_out,
        planes,
        func or region_abs_mask,
    )


def squaremask(
    clip: vs.VideoNode,
    width: int,
    height: int,
    offset_x: int,
    offset_y: int,
    invert: bool = False,
    force_gray: bool = True,
    planes: Planes = None,
    func: FuncExcept | None = None,
) -> vs.VideoNode:
    """
    Create a square used for simple masking.

    This is a fast and simple mask that's useful for very rough and simple masking.

    Args:
        clip: The clip to process.
        width: The width of the square. This must be less than clip.width - offset_x.
        height: The height of the square. This must be less than clip.height - offset_y.
        offset_x: The location of the square, offset from the left side of the frame.
        offset_y: The location of the square, offset from the top of the frame.
        invert: Invert the mask. This means everything *but* the defined square will be masked. Default: False.
        force_gray: Whether to force using GRAY format or clip format.
        planes: Which planes to process.
        func: Function returned for custom error handling. This should only be set by VS package developers.

    Returns:
        A mask in the shape of a square.
    """
    func = func or squaremask

    mask_format = (
        clip.format.replace(color_family=vs.GRAY, subsampling_w=0, subsampling_h=0) if force_gray else clip.format
    )

    if offset_x + width > clip.width or offset_y + height > clip.height:
        raise CustomValueError("mask exceeds clip size!", func)

    base_clip = vs.core.std.BlankClip(
        clip, None, None, mask_format.id, 1, color=get_lowest_values(mask_format, ColorRange.FULL), keep=True
    )

    replaces = ("mask_max", "x") if not invert else ("x", "mask_max")
    mask = region_abs_mask(base_clip, width, height, offset_x, offset_y, *replaces, planes, func)

    if clip.num_frames == 1:
        return mask

    return core.std.Loop(mask, clip.num_frames)


def replace_squaremask(
    clipa: vs.VideoNode,
    clipb: vs.VideoNode,
    mask_params: tuple[int, int, int, int],
    ranges: FrameRangeN | FrameRangesN | None = None,
    blur_sigma: int | float | None = None,
    invert: bool = False,
    func: FuncExcept | None = None,
    show_mask: bool = False,
) -> vs.VideoNode:
    """
    Replace an area of the frame with another clip using a simple square mask.

    This is a convenience wrapper merging square masking and framerange replacing functionalities
    into one function, along with additional utilities such as blurring.

    Args:
        clipa: Base clip to process.
        clipb: Clip to mask on top of `clipa`.
        mask_params: Parameters passed to `squaremask`. Expects a tuple of (width, height, offset_x, offset_y).
        ranges: Frameranges to replace with the masked clip. If `None`, replaces the entire clip. Default: None.
        blur_sigma: Post-blurring of the mask to help hide hard edges.
            If you pass an int, a [box_blur][vsrgtools.box_blur] will be used.
            Passing a float will use a [gauss_blur][vsrgtools.gauss_blur] instead.
            Default: None.
        invert: Invert the mask. This means everything *but* the defined square will be masked. Default: False.
        func: Function returned for custom error handling. This should only be set by VS package developers. Default:
            [squaremask][vsmasktools.squaremask].
        show_mask: Return the mask instead of the masked clip.

    Returns:
        Clip with a squaremask applied, and optionally set to specific frameranges.
    """
    func = func or replace_squaremask

    mask = squaremask(clipb[0], *mask_params, invert, func=func)

    if isinstance(blur_sigma, int):
        mask = box_blur(mask, blur_sigma)
    elif isinstance(blur_sigma, float):
        mask = gauss_blur(mask, blur_sigma)

    mask = core.std.Loop(mask, clipa.num_frames)

    if show_mask:
        return mask

    merge = clipa.std.MaskedMerge(clipb, mask)

    ranges = normalize_ranges(clipa, ranges)

    if len(ranges) == 1 and ranges[0] == (0, clipa.num_frames - 1):
        return merge

    return replace_ranges(clipa, merge, ranges)


def freeze_replace_squaremask(
    mask: vs.VideoNode,
    insert: vs.VideoNode,
    mask_params: tuple[int, int, int, int],
    frame: int,
    frame_range: tuple[int, int],
) -> vs.VideoNode:
    start, end = frame_range

    masked_insert = replace_squaremask(mask[frame], insert[frame], mask_params)

    return insert_clip(mask, masked_insert * (end - start + 1), start)


@overload
def normalize_mask(mask: vs.VideoNode, clip: vs.VideoNode, *, func: FuncExcept | None = None) -> vs.VideoNode: ...


@overload
def normalize_mask(
    mask: Callable[[vs.VideoNode, vs.VideoNode], vs.VideoNode],
    clip: vs.VideoNode,
    ref: vs.VideoNode,
    *,
    func: FuncExcept | None = None,
) -> vs.VideoNode: ...


@overload
def normalize_mask(
    mask: EdgeDetectLike | RidgeDetectLike,
    clip: vs.VideoNode,
    *,
    ridge: bool = ...,
    func: FuncExcept | None = None,
    **kwargs: Any,
) -> vs.VideoNode: ...


@overload
def normalize_mask(
    mask: GeneralMask, clip: vs.VideoNode, ref: vs.VideoNode, *, func: FuncExcept | None = None
) -> vs.VideoNode: ...


@overload
def normalize_mask(
    mask: MaskLike,
    clip: vs.VideoNode,
    ref: vs.VideoNode | None = ...,
    *,
    ridge: bool = ...,
    func: FuncExcept | None = None,
    **kwargs: Any,
) -> vs.VideoNode: ...


def normalize_mask(
    mask: MaskLike,
    clip: vs.VideoNode,
    ref: vs.VideoNode | None = None,
    *,
    ridge: bool = False,
    func: FuncExcept | None = None,
    **kwargs: Any,
) -> vs.VideoNode:
    """
    Normalize any mask type to match the format and range of the input clip.

    Args:
        mask: The mask to normalize. Can be:

               - A `VideoNode` representing a precomputed mask.
               - A callable that takes `(clip, ref)` and returns a `VideoNode`.
               - An `EdgeDetect` or `RidgeDetect` instance or type.
               - A `GeneralMask` instance.
        clip: The clip to which the output mask will be normalized.
        ref: A reference clip required by certain mask functions or classes.
        ridge: If `True` and `mask` is a `RidgeDetect` instance, generate a ridge mask instead of an edge mask.
            Defaults to `False`.
        func: Function returned for custom error handling. This should only be set by VS package developers.
        **kwargs: Additional keyword arguments passed to the edge/ridge detection methods.

    Raises:
        CustomValueError: If `mask` is a callable that requires a reference and `ref` is not provided.

    Returns:
        A mask with the same format as `clip`.
    """
    func = func or normalize_mask

    if isinstance(mask, (str, type)):
        return normalize_mask(EdgeDetect.ensure_obj(mask, func), clip, ref, ridge=ridge, func=func, **kwargs)

    if isinstance(mask, EdgeDetect):
        if ridge and isinstance(mask, RidgeDetect):
            cmask = mask.ridgemask(clip, **kwargs)
        else:
            cmask = mask.edgemask(clip, **kwargs)
    elif isinstance(mask, GeneralMask):
        cmask = mask.get_mask(clip, ref)
    elif callable(mask):
        if ref is None:
            raise CustomValueError("This mask function requires a ref to be specified!", func)

        cmask = mask(clip, ref)
    else:
        cmask = mask

    return depth(cmask, clip, range_in=ColorRange.FULL, range_out=ColorRange.FULL)


class RektPartial[**P, R]:
    """
    Class decorator that wraps the [rekt_partial][vsmasktools.utils.rekt_partial] function
    and extends its functionality.

    It is not meant to be used directly.
    """

    def __init__(self, rekt_partial: Callable[P, R]) -> None:
        self._func = rekt_partial

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self._func(*args, **kwargs)

    def rel(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self._func(*args, **kwargs)

    def abs(
        self,
        clip: vs.VideoNode,
        func: Callable[Concatenate[vs.VideoNode, ...], vs.VideoNode],
        width: int,
        height: int,
        offset_x: int = 0,
        offset_y: int = 0,
        *args: Any,
        **kwargs: Any,
    ) -> vs.VideoNode:
        """
        Creates a rectangular mask to apply fixes only within the masked area,
        significantly speeding up filters like anti-aliasing and scaling.

        Args:
            clip: The source video clip to which the mask will be applied.
            func: The function to be applied within the masked area.
            width: The width of the rectangular mask.
            height: The height of the rectangular mask.
            offset_x: The horizontal offset of the mask from the top-left corner, defaults to 0.
            offset_y: The vertical offset of the mask from the top-left corner, defaults to 0.

        Returns:
            A new clip with the applied mask.
        """
        nargs = (clip, func, offset_x, clip.width - width - offset_x, offset_y, clip.height - height - offset_y)
        return self._func(*nargs, *args, **kwargs)  # type: ignore[return-value, arg-type]


@RektPartial
def rekt_partial(
    clip: vs.VideoNode,
    func: Callable[Concatenate[vs.VideoNode, ...], vs.VideoNode],
    left: int = 0,
    right: int = 0,
    top: int = 0,
    bottom: int = 0,
    *args: Any,
    **kwargs: Any,
) -> vs.VideoNode:
    """
    Creates a rectangular mask to apply fixes only within the masked area,
    significantly speeding up filters like anti-aliasing and scaling.

    Args:
        clip: The source video clip to which the mask will be applied.
        func: The function to be applied within the masked area.
        left: The left boundary of the mask, defaults to 0.
        right: The right boundary of the mask, defaults to 0.
        top: The top boundary of the mask, defaults to 0.
        bottom: The bottom boundary of the mask, defaults to 0.

    Returns:
        A new clip with the applied mask.
    """

    def _filtered_func(clip: vs.VideoNode, *args: Any, **kwargs: Any) -> vs.VideoNode:
        return func(clip, *args, **kwargs)

    if left == top == right == bottom == 0:
        return _filtered_func(clip, *args, **kwargs)

    cropped = clip.std.Crop(left, right, top, bottom)

    filtered = _filtered_func(cropped, *args, **kwargs)

    check_ref_clip(cropped, filtered, rekt_partial._func)

    filtered = core.std.AddBorders(filtered, left, right, top, bottom)

    ratio_w, ratio_h = 2**clip.format.subsampling_w, 2**clip.format.subsampling_h

    vals = list(
        filter(
            None,
            [
                ("X {left} >= " if left else None),
                ("X {right} < " if right else None),
                ("Y {top} >= " if top else None),
                ("Y {bottom} < " if bottom else None),
            ],
        )
    )

    return norm_expr(
        [clip, filtered],
        [*vals, ["and"] * (len(vals) - 1), "y x ?"],
        left=[left, left / ratio_w],
        right=[clip.width - right, (clip.width - right) / ratio_w],
        top=[top, top / ratio_h],
        bottom=[clip.height - bottom, (clip.height - bottom) / ratio_h],
        func=rekt_partial._func,
    )
