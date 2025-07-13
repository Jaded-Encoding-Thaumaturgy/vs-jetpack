from __future__ import annotations

from typing import Any, Sequence, SupportsFloat, cast

from jetpytools import CustomNotImplementedError, CustomRuntimeError, CustomStrEnum

from vsaa import BWDIF, NNEDI3, Deinterlacer
from vsexprtools import norm_expr
from vsmasktools import FDoG, GenericMaskT, Morpho, adg_mask, normalize_mask, strength_zones_mask
from vsrgtools import MeanMode, gauss_blur, repair
from vsscale import DPIR
from vstools import (
    Align,
    ConstantFormatVideoNode,
    FieldBased,
    FrameRangeN,
    FrameRangesN,
    FunctionUtil,
    PlanesT,
    check_progressive,
    check_variable,
    check_variable_format,
    core,
    depth,
    fallback,
    get_plane_sizes,
    get_y,
    join,
    normalize_ranges,
    normalize_seq,
    padder,
    plane,
    replace_ranges,
    shift_clip_multi,
    vs,
)

__all__ = ["deblock_qed", "dpir", "dpir_mask", "mpeg2stinx"]

_StrengthT = SupportsFloat | vs.VideoNode | None


class dpir(CustomStrEnum):  # noqa: N801
    DEBLOCK = cast("dpir", "deblock")
    DENOISE = cast("dpir", "denoise")

    def __call__(
        self,
        clip: vs.VideoNode,
        strength: _StrengthT | Sequence[_StrengthT] = 10,
        zones: Sequence[tuple[FrameRangeN | FrameRangesN, _StrengthT]] | None = None,
        **kwargs: Any,
    ) -> ConstantFormatVideoNode:
        """
        Deep Plug-and-Play Image Restoration

        Args:
            clip: Clip to process.
            strength: Threshold (8-bit scale) strength for deblocking/denoising.
                This can be:

                  - A single value or VideoNode applied to all planes.
                  - A sequence of values VideoNodes to specify per-plane thresholds.

                If a VideoNode is used, it must be in GRAY8, GRAYH, or GRAYS format,
                with pixel values representing the 8-bit thresholds.

            zones: Apply different strength in specified zones.
            **kwargs: Additional arguments to be passed to `vsscale.DPIR`.
        """
        func = "dpir." + str(self.value)

        assert check_variable_format(clip, func)

        if isinstance(strength, Sequence):
            if len(strength) == 1:
                return join(self.__call__(get_y(clip), strength[0], zones, **kwargs), clip)

            if len(strength) == 2:
                return join(
                    self.__call__(get_y(clip), strength[0], zones, **kwargs),
                    self.__call__(clip, strength[1], zones, **kwargs),
                )

            if len(strength) == 3:
                return join(
                    [self.__call__(plane(clip, i), s, zones, **kwargs) for i, s in enumerate(strength)],
                    clip.format.color_family,
                )

            raise CustomRuntimeError

        if not strength:
            return clip

        if not isinstance(strength, vs.VideoNode):
            base_strength = clip.std.BlankClip(format=vs.GRAYH, color=float(strength))
        else:
            base_strength = strength

        strength = strength_zones_mask(base_strength, zones, vs.GRAYH, clip.num_frames)

        if self.value == "deblock":
            dpired = DPIR.DrunetDeblock(strength, **kwargs).scale(clip)
        elif self.value == "denoise":
            dpired = DPIR.DrunetDenoise(strength, **kwargs).scale(clip)
        else:
            raise CustomNotImplementedError(func=func, reason=self.value)

        if zones is not None:
            no_dpir_zones = list[tuple[int, int]]()

            for r, s in zones:
                if s is None or (not isinstance(s, vs.VideoNode) and float(s) == 0):
                    no_dpir_zones.extend(normalize_ranges(clip, r))

            dpired = replace_ranges(dpired, clip, no_dpir_zones)

        return dpired


def dpir_mask(
    clip: vs.VideoNode,
    low: float = 5,
    high: float = 10,
    lines: float | None = None,
    luma_scaling: float = 12,
    linemask: GenericMaskT | bool = True,
    relative: bool = False,
) -> vs.VideoNode:
    y = depth(get_y(clip), 32)

    if linemask is True:
        linemask = FDoG

    mask = adg_mask(y, luma_scaling, relative, func=dpir_mask)

    if relative:
        mask = gauss_blur(mask, 1.5)

    mask = norm_expr(mask, f"{high} 255 / x {low} 255 / * -", func=dpir_mask)

    if linemask:
        lines = fallback(lines, high)
        linemask = normalize_mask(linemask, y)

        lines_clip = mask.std.BlankClip(color=lines / 255)

        mask = mask.std.MaskedMerge(lines_clip, linemask)

    return mask


def deblock_qed(
    clip: vs.VideoNode,
    quant_edge: int = 24,
    quant_inner: int = 26,
    alpha_edge: int = 1,
    beta_edge: int = 2,
    alpha_inner: int = 1,
    beta_inner: int = 2,
    chroma_mode: int = 0,
    align: Align = Align.TOP_LEFT,
    planes: PlanesT = None,
) -> ConstantFormatVideoNode:
    """
    A postprocessed Deblock: Uses full frequencies of Deblock's changes on block borders,
    but DCT-lowpassed changes on block interiours.

    Args:
        clip: Clip to process.
        quant_edge: Strength of block edge deblocking.
        quant_inner: Strength of block internal deblocking.
        alpha_edge: Halfway "sensitivity" and halfway a strength modifier for borders.
        beta_edge: "Sensitivity to detect blocking" for borders.
        alpha_inner: Halfway "sensitivity" and halfway a strength modifier for block interiors.
        beta_inner: "Sensitivity to detect blocking" for block interiors.
        chroma_mode: Chroma deblocking behaviour.

               - 0 = use proposed method for chroma deblocking
               - 1 = directly use chroma deblock from the normal Deblock
               - 2 = directly use chroma deblock from the strong Deblock

        align: Where to align the blocks for eventual padding.
        planes: What planes to process.

    Returns:
        Deblocked clip
    """
    func = FunctionUtil(clip, deblock_qed, planes)

    if not func.chroma:
        chroma_mode = 0

    with padder.ctx(8, align=align) as p8:
        clip = p8.MIRROR(func.work_clip)

        block = padder.COLOR(
            core.std.BlankClip(
                clip,
                width=6,
                height=6,
                length=1,
                color=0,
                format=func.work_clip.format.replace(color_family=vs.GRAY, subsampling_w=0, subsampling_h=0).id,
            ),
            1,
            1,
            1,
            1,
            True,
        )
        block = core.std.StackHorizontal([block] * (clip.width // block.width))
        block = core.std.StackVertical([block] * (clip.height // block.height))

        if func.chroma:
            blockc = block.std.CropAbs(*get_plane_sizes(clip, 1))
            block = join(block, blockc, blockc)

        block = block * clip.num_frames

        normal, strong = (
            clip.deblock.Deblock(quant_edge, alpha_edge, beta_edge, func.norm_planes if chroma_mode < 2 else 0),
            clip.deblock.Deblock(quant_inner, alpha_inner, beta_inner, func.norm_planes if chroma_mode != 1 else 0),
        )

        normal_d2, strong_d2 = (
            norm_expr([clip, dclip, block], "z x y - 0 ? neutral +", planes) for dclip in (normal, strong)
        )

        with padder.ctx(16, align=align) as p16:
            strong_d2 = p16.CROP(
                norm_expr(p16.MIRROR(strong_d2), "x neutral - 1.01 * neutral +", planes, func=func.func).dctf.DCTFilter(
                    [1, 1, 0, 0, 0, 0, 0, 0], planes
                )
            )

        strong_d4 = norm_expr([strong_d2, normal_d2], "y neutral = x y ?", planes, func=func.func)
        deblocked = clip.std.MakeDiff(strong_d4, planes)

        if func.chroma and chroma_mode:
            deblocked = join([deblocked, strong if chroma_mode == 2 else normal])

        deblocked = p8.CROP(deblocked)

    return func.return_clip(deblocked)


def mpeg2stinx(
    clip: vs.VideoNode,
    bobber: Deinterlacer = NNEDI3(),
    tff: bool = True,
    mask: bool = True,
    radius: int | tuple[int, int] = 2,
    limit: float | None = 1.0,
) -> ConstantFormatVideoNode:
    """
    This filter is designed to eliminate certain combing-like compression artifacts that show up all too often
    in hard-telecined MPEG-2 encodes, and works to a smaller extent on bitrate-starved hard-telecined AVC as well.
    General artifact removal is better accomplished with actual denoisers.

    Args:
        clip: Clip to process.
        tff: The field order.
        mask: Whether to use BWDIF motion masking.
        bobber: Callable to use in place of the internal deinterlacing filter.
        radius: x, y radius of min-max clipping (i.e. repair) to remove artifacts.
        limit: If specified, temporal limiting is used, where the changes by crossfieldrepair are limited to this times
            the difference between the current frame and its neighbours.

    Returns:
        Clip with cross-field noise reduced.
    """

    def _crossfield_repair(clip: vs.VideoNode, bobbed: vs.VideoNode) -> vs.VideoNode:
        clip = core.std.Interleave([clip] * 2)

        if sw == 1 and sh == 1:
            repaired = repair(clip, bobbed, 1)
        else:
            inpand, expand = Morpho.inpand(bobbed, sw, sh), Morpho.expand(bobbed, sw, sh)
            repaired = MeanMode.MEDIAN([clip, inpand, expand])

        return repaired.std.SeparateFields(tff).std.SelectEvery(4, (2, 1)).std.DoubleWeave(tff)[::2]

    def _temporal_limit(src: vs.VideoNode, flt: vs.VideoNode, adj: vs.VideoNode | None) -> vs.VideoNode:
        if limit is None:
            return flt

        assert adj

        diff = norm_expr([core.std.Interleave([src] * 2), adj], "x y - abs", func=mpeg2stinx).std.SeparateFields(tff)
        diff = MeanMode.MINIMUM([diff.std.SelectEvery(4, (0, 1)), diff.std.SelectEvery(4, (2, 3))])
        diff = Morpho.expand(diff, sw=2, sh=1).std.DoubleWeave(tff)[::2]

        return norm_expr([flt, src, diff], "z {limit} * LIM! x y LIM@ - y LIM@ + clip", limit=limit, func=mpeg2stinx)

    def _bobfunc(clip: vs.VideoNode) -> vs.VideoNode:
        bobbed = bobber.bob(clip)

        if mask:
            bobbed = BWDIF(tff=tff, edeint=bobbed).bob(clip)

        return bobbed

    assert check_variable(clip, mpeg2stinx)
    assert check_progressive(clip, mpeg2stinx)

    sw, sh = normalize_seq(radius, 2)
    bobber = bobber.copy(tff=tff)

    if limit is not None:
        adjs = shift_clip_multi(clip)
        adjs.pop(1)
        adj = core.std.Interleave(adjs)
    else:
        adj = None

    fixed1 = _temporal_limit(clip, _crossfield_repair(clip, _bobfunc(clip)), adj)
    fixed2 = _temporal_limit(fixed1, _crossfield_repair(fixed1, _bobfunc(fixed1)), adj)

    return core.std.SetFieldBased(fixed1.std.Merge(fixed2), FieldBased.PROGRESSIVE)
