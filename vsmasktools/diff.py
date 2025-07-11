from __future__ import annotations

import warnings
from typing import Any, TypeAlias, Union, overload

from vsexprtools import ExprOp, norm_expr
from vskernels import Bilinear, Catrom, Kernel, KernelLike
from vsrgtools import bilateral, gauss_blur, remove_grain
from vsrgtools.rgtools import RemoveGrain
from vstools import (
    ColorRange,
    ConstantFormatVideoNode,
    CustomValueError,
    FuncExceptT,
    KwargsT,
    VSFunctionNoArgs,
    check_variable,
    depth,
    get_w,
    get_y,
    insert_clip,
    iterate,
    vs,
)

from .edge import EdgeDetect, EdgeDetectT, ExLaplacian4
from .morpho import Morpho
from .types import XxpandMode

__all__ = ["based_diff_mask", "credit_mask", "diff_creditless", "diff_creditless_oped", "diff_rescale"]


def diff_rescale(
    clip: vs.VideoNode,
    height: int,
    kernel: KernelLike = Catrom,
    thr: float = 0.216,
    expand: int = 2,
    func: FuncExceptT | None = None,
) -> ConstantFormatVideoNode:
    return based_diff_mask(clip, height, kernel, thr, expand=2 + expand, func=func)


def diff_creditless(
    credit_clip: vs.VideoNode,
    nc_clip: vs.VideoNode,
    thr: float = 0.01,
    start_frame: int = 0,
    expand: int = 2,
    *,
    prefilter: bool | int = False,
    edgemask: EdgeDetectT = ExLaplacian4,
    ep_clip: vs.VideoNode | None = None,
    func: FuncExceptT | None = None,
    **kwargs: Any,
) -> ConstantFormatVideoNode:
    mask = based_diff_mask(
        credit_clip,
        nc_clip,
        thr=thr,
        prefilter=prefilter,
        postfilter=0,
        ampl=EdgeDetect.ensure_obj(edgemask, func),
        expand=2 + expand,
        func=func,
    )

    if not ep_clip or ep_clip.num_frames == mask.num_frames:
        return mask

    return insert_clip(ep_clip.std.BlankClip(format=mask.format.id, keep=True), mask, start_frame)


def diff_creditless_oped(
    ep: vs.VideoNode,
    ncop: vs.VideoNode | None,
    nced: vs.VideoNode | None,
    thr: float = 0.1,
    opstart: int | None = None,
    opend: int | None = None,
    edstart: int | None = None,
    edend: int | None = None,
    func: FuncExceptT | None = None,
    **kwargs: Any,
) -> ConstantFormatVideoNode:
    func = func or diff_creditless_oped

    op_mask = ed_mask = None

    kwargs |= KwargsT(expand=4, prefilter=False, func=func, ep_clip=ep) | kwargs

    if opstart is not None and opend is not None and ncop is not None:
        op_mask = diff_creditless(ep[opstart : opend + 1], ncop[: opend - opstart + 1], thr, opstart, **kwargs)

    if edstart is not None and edend is not None and nced is not None:
        ed_mask = diff_creditless(ep[edstart : edend + 1], nced[: edend - edstart + 1], thr, edstart, **kwargs)

    if op_mask and ed_mask:
        return ExprOp.ADD.combine(op_mask, ed_mask)

    if op_mask:
        return op_mask

    if ed_mask:
        return ed_mask

    raise CustomValueError('You must specify one or both of ("opstart", "opend"), ("edstart", "edend")', func)


def credit_mask(
    clip: vs.VideoNode,
    ref: vs.VideoNode,
    thr: float,
    blur: float | None = 1.65,
    prefilter: bool | int = 5,
    expand: int = 8,
) -> ConstantFormatVideoNode:
    warnings.warn(
        "credit_mask: Function is deprecated and will be removed in a later version! Use based_diff_mask",
        DeprecationWarning,
    )

    if blur is not None:
        clip, ref = gauss_blur(clip, blur), gauss_blur(ref, blur)

    credit_mask = based_diff_mask(clip, ref, thr=thr, prefilter=prefilter, postfilter=0, ampl=ExLaplacian4, expand=4)

    credit_mask = Morpho.erosion(credit_mask, iterations=6)
    credit_mask = iterate(credit_mask, lambda x: x.std.Minimum().std.Maximum(), 8)

    if expand:
        credit_mask = Morpho.dilation(credit_mask, iterations=expand)

    credit_mask = Morpho.inflate(credit_mask, iterations=3)

    return credit_mask


Count: TypeAlias = int


@overload
def based_diff_mask(
    clip: vs.VideoNode,
    ref: vs.VideoNode,
    /,
    *,
    thr: float = 0.216,
    prefilter: int | KwargsT | bool | VSFunctionNoArgs[vs.VideoNode, ConstantFormatVideoNode] = False,
    postfilter: Union[
        int,
        tuple[Count, RemoveGrain.Mode],
        list[tuple[Count, RemoveGrain.Mode]],
        VSFunctionNoArgs[vs.VideoNode, ConstantFormatVideoNode],
    ] = 2,
    ampl: str | type[EdgeDetect] | EdgeDetect = ...,
    expand: int = 4,
    func: FuncExceptT | None = None,
) -> ConstantFormatVideoNode: ...


@overload
def based_diff_mask(
    clip: vs.VideoNode,
    height: int,
    kernel: KernelLike,
    /,
    thr: float = 0.216,
    prefilter: int | KwargsT | bool | VSFunctionNoArgs[vs.VideoNode, ConstantFormatVideoNode] = False,
    postfilter: Union[
        int,
        tuple[Count, RemoveGrain.Mode],
        list[tuple[Count, RemoveGrain.Mode]],
        VSFunctionNoArgs[vs.VideoNode, ConstantFormatVideoNode],
    ] = 2,
    ampl: str | type[EdgeDetect] | EdgeDetect = ...,
    expand: int = 4,
    func: FuncExceptT | None = None,
) -> ConstantFormatVideoNode: ...


def based_diff_mask(
    clip: vs.VideoNode,
    ref_or_height: vs.VideoNode | int,
    kernel: KernelLike | None = None,
    /,
    thr: float = 0.216,
    prefilter: int | KwargsT | bool | VSFunctionNoArgs[vs.VideoNode, ConstantFormatVideoNode] = False,
    postfilter: Union[
        int,
        tuple[Count, RemoveGrain.Mode],
        list[tuple[Count, RemoveGrain.Mode]],
        VSFunctionNoArgs[vs.VideoNode, ConstantFormatVideoNode],
    ] = 2,
    ampl: str | type[EdgeDetect] | EdgeDetect = "x yrange_max / 2 4 pow * {thr} < 0 1 ? yrange_max *",
    expand: int = 4,
    func: FuncExceptT | None = None,
) -> ConstantFormatVideoNode:
    """
    Make a difference mask between a clean source and a reference clip with additionnal pre and post processing

    Args:
        clip: Source clip
        ref_or_height: Reference clip or height to be descaled to.
        kernel: Kernel used for descaling and rescaling
        thr: Threshold of the amplification expr, defaults to 0.216
        prefilter: Filter applied before extracting the difference between clip and ref:

               - int -> equivalent of number of taps used in the bilateral call applied to the clips
               - True -> 5 taps
               - KwargsT -> Arguments passed to the bilateral function
        postfilter: Filter applied to the difference clip. Default is RemoveGrainMode.MINMAX_AROUND2 applied twice.
        ampl: Amplification expression.
        expand: Additional expand radius applied to the mask, defaults to 4

    Returns:
        Generated mask
    """
    func = func or based_diff_mask

    assert check_variable(clip, func)

    if isinstance(ref_or_height, vs.VideoNode):
        ref = ref_or_height
    else:
        clip = get_y(clip)

        assert kernel is not None

        kernel = Kernel.ensure_obj(kernel, func)

        ref = kernel.descale(clip, get_w(ref_or_height), ref_or_height)
        ref = kernel.scale(ref, clip.width, clip.height)

    assert check_variable(ref, func)

    if clip.format.num_planes != ref.format.num_planes:
        clip, ref = get_y(clip), get_y(ref)

    if prefilter:
        if callable(prefilter):
            clip, ref = prefilter(clip), prefilter(ref)
        else:
            if isinstance(prefilter, int):
                sigma = 5 if prefilter is True else prefilter
                kwargs = KwargsT(sigmaS=((sigma**2 - 1) / 12) ** 0.5, sigmaR=sigma / 10)
            else:
                kwargs = prefilter

            clip, ref = bilateral(clip, **kwargs), bilateral(ref, **kwargs)

    ref = depth(ref, clip)

    dst_fmt = clip.format.replace(subsampling_w=0, subsampling_h=0)
    diff_fmt = dst_fmt.replace(color_family=vs.GRAY)

    mask = ExprOp.mae(dst_fmt)(
        (Bilinear().resample(c, dst_fmt) for c in [clip, ref]), format=diff_fmt, split_planes=True
    )
    mask = ColorRange.FULL.apply(mask)

    if postfilter:
        if isinstance(postfilter, int):
            mask = iterate(mask, remove_grain, postfilter, remove_grain.Mode.MINMAX_AROUND2)
        elif isinstance(postfilter, tuple):
            mask = iterate(mask, remove_grain, postfilter[0], postfilter[1])
        elif isinstance(postfilter, list):
            mask = mask
            for count, rgmode in postfilter:
                mask = iterate(mask, remove_grain, count, rgmode)
        else:
            mask = postfilter(mask)

    if isinstance(ampl, str):
        mask = norm_expr(mask, ampl.format(thr=thr), func=func)
    else:
        mask = EdgeDetect.ensure_obj(ampl, func).edgemask(mask, lthr=thr, hthr=thr)

    if expand:
        mask = Morpho.expand(mask, expand, mode=XxpandMode.ELLIPSE, func=func)

    return mask
