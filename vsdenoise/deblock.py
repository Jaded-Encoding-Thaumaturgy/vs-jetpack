from __future__ import annotations

from typing import Any, Sequence, SupportsFloat, cast

from jetpytools import CustomNotImplementedError, CustomRuntimeError, CustomStrEnum, fallback, normalize_seq

from vsaa import BWDIF, NNEDI3, Deinterlacer
from vsexprtools import norm_expr
from vskernels import Box, Point
from vsmasktools import FDoG, MaskLike, Morpho, adg_mask, normalize_mask, strength_zones_mask
from vsrgtools import MeanMode, gauss_blur, repair
from vsscale import DPIR
from vstools import (
    FieldBased,
    FrameRangeN,
    FrameRangesN,
    Planes,
    UnsupportedColorFamilyError,
    check_progressive,
    core,
    depth,
    get_y,
    join,
    normalize_param_planes,
    normalize_planes,
    normalize_ranges,
    plane,
    replace_ranges,
    shift_clip_multi,
    vs,
)

__all__ = ["deblock_qed", "dpir", "dpir_mask", "mpeg2stinx"]

_StrengthT = SupportsFloat | vs.VideoNode | None


class dpir(CustomStrEnum):  # noqa: N801
    """Deep Plug-and-Play Image Restoration."""

    DEBLOCK = cast("dpir", "deblock")
    """DPIR model for deblocking."""

    DENOISE = cast("dpir", "denoise")
    """DPIR model for denoising."""

    def __call__(
        self,
        clip: vs.VideoNode,
        strength: _StrengthT | Sequence[_StrengthT] = 10,
        zones: Sequence[tuple[FrameRangeN | FrameRangesN, _StrengthT]] | None = None,
        planes: Planes = None,
        **kwargs: Any,
    ) -> vs.VideoNode:
        """
        Deep Plug-and-Play Image Restoration.

        Example:
            Apply DPIR to an entire clip with a global strength of 15,
            a strength of 20 in the frame range (10, 20),
            a strength of 5 in the frame ranges (500, 1000) and (1500, 2500),
            restricted to the chroma planes only,
            and using the TRT_RTX backend:

            ```py
            import vsmlrt

            clip = dpir.DEBLOCK(
                clip,
                15,
                [((10, 20), 20), ([(500, 1000), (1500, 2500)], 5)],
                planes=[1, 2],
                backend=vsmlrt.Backend.TRT_RTX,
            )
            ```

        Notes:
            - DPIR requires RGB or GRAY input (automatically upsampled if YUV).
            - The default chroma upscaler is Catrom.
              Users may substitute a higher-quality upscaler if needed.

        Args:
            clip: Clip to process.
            strength: Threshold (8-bit scale) strength for deblocking/denoising.
                This can be:

                  - A single value or VideoNode applied to all planes.
                  - A sequence of values or VideoNodes to specify per-plane thresholds.

                If a VideoNode is used, it must be in GRAY8, GRAYH, or GRAYS format,
                with pixel values representing the 8-bit thresholds.

            zones: Apply different strength values to specific frame ranges.
            **kwargs: Additional arguments to be passed to [vsscale.DPIR][].
        """
        func = "dpir." + str(self.value)

        planes = normalize_planes(clip, planes)

        if isinstance(strength, Sequence):
            UnsupportedColorFamilyError.check(
                clip, vs.YUV, func, "Input clip must be {correct} when passing a sequence of strength."
            )

            if len(strength) == 1:
                return join(self.__call__(get_y(clip), strength[0], zones, **kwargs), clip)

            if len(strength) == 2:
                plane_0 = get_y(clip)

                return join(
                    {
                        None: clip,
                        tuple(planes): self.__call__(clip, strength[1], zones, planes, **kwargs),
                        0: self.__call__(plane_0, strength[0], zones, planes, **kwargs),
                    }
                )

            if len(strength) == 3:
                strength = normalize_param_planes(clip, strength, planes, 0)

                return join(
                    [self.__call__(plane(clip, i), s, zones, **kwargs) for i, s in enumerate(strength)],
                    clip.format.color_family,
                )

            raise CustomRuntimeError

        if not strength or not planes:
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

            out = replace_ranges(dpired, clip, no_dpir_zones)
        else:
            out = dpired

        if planes != normalize_planes(clip, None):
            out = join({None: clip, tuple(planes): dpired}, clip.format.color_family, prop_src=clip)

        return out


def dpir_mask(
    clip: vs.VideoNode,
    low: float = 5,
    high: float = 10,
    lines: float | None = None,
    luma_scaling: float = 12,
    linemask: MaskLike | bool = True,
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
    quant: tuple[int | None, int | None] = (24, 26),
    alpha: tuple[int | None, int | None] = (1, 1),
    beta: tuple[int | None, int | None] = (2, 2),
    chroma_mode: int = 0,
    planes: Planes = None,
) -> vs.VideoNode:
    """
    A post-processed deblock. Uses full frequencies of Deblock's changes on block borders, but DCT-lowpassed changes on
    block interiors. Designed to provide 8x8 deblocking sensitive to the amount of blocking in the source, compared to
    other deblockers which apply a uniform deblocking across every frame.

    Args:
        clip: Clip to process.
        quant: Strength of the deblocking. Tuple for (border, interior) values.
        alpha: Both a sensitivity and strength modifier. Tuple for (border, interior) values.
        beta: Sensitivity to detect blocking. Tuple for (border, interior) values.
        chroma_mode: Chroma deblocking behaviour.

               - 0 = Use proposed method for chroma deblocking.
               - 1 = Directly use chroma deblock from the normal deblock.
               - 2 = Directly use chroma deblock from the strong deblock.

        planes: Planes to process.

    Returns:
        Deblocked clip
    """
    fieldbased = FieldBased.from_video(clip, func=deblock_qed)
    planes_pp = 0 if chroma_mode else planes

    if fieldbased.is_inter():
        clip = Point().scale(clip.std.SeparateFields(fieldbased.is_tff()), height=clip.height)

    normal, strong = (
        clip.deblock.Deblock(quant[0], alpha[0], beta[0], planes),
        clip.deblock.Deblock(quant[1], alpha[1], beta[1], planes),
    )

    mask = norm_expr(
        clip[0],
        "X 8 % 7 % Y 8 % 7 % and 0 255 ?",
        planes_pp,
        clip.format.replace(sample_type=vs.SampleType.INTEGER, bits_per_sample=8),
        func=deblock_qed,
    )

    strong_diff = norm_expr([clip, strong, mask], "z x y - 1.01 * neutral + neutral ?", planes_pp, func=deblock_qed)
    strong_pp = strong_diff.dctf.DCTFilter([1, 1, 0, 0, 0, 0, 0, 0], planes_pp)
    deblocked = norm_expr([clip, normal, strong_pp, mask], "a y x z neutral - - ?", planes_pp, func=deblock_qed)

    if clip.format.color_family is not vs.GRAY:
        if chroma_mode == 1:
            deblocked = join(deblocked, normal)
        if chroma_mode == 2:
            deblocked = join(deblocked, strong)

    if fieldbased.is_inter():
        from vsdeinterlace import weave

        deblocked = weave(Box().scale(deblocked, height=clip.height // 2), fieldbased, deblock_qed)

    return deblocked


def mpeg2stinx(
    clip: vs.VideoNode,
    bobber: Deinterlacer = NNEDI3(),
    tff: bool = True,
    mask: bool = True,
    radius: int | tuple[int, int] = 2,
    limit: float | None = 1.0,
) -> vs.VideoNode:
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
    from vsdeinterlace import weave

    def _crossfield_repair(clip: vs.VideoNode, bobbed: vs.VideoNode) -> vs.VideoNode:
        clip = core.std.Interleave([clip, clip])

        if sw == sh == 1:
            repaired = repair(clip, bobbed, 1)
        else:
            inpand, expand = (
                Morpho.inpand(bobbed, sw, sh, func=mpeg2stinx),
                Morpho.expand(bobbed, sw, sh, func=mpeg2stinx),
            )
            repaired = norm_expr([clip, inpand, expand], "x y z clip", func=mpeg2stinx)

        return weave(repaired.std.SeparateFields(tff).std.SelectEvery(4, (2, 1)), tff, func=mpeg2stinx)

    def _temporal_limit(src: vs.VideoNode, flt: vs.VideoNode, adj: vs.VideoNode | None) -> vs.VideoNode:
        if limit is None:
            return flt

        assert adj

        diff = norm_expr([core.std.Interleave([src, src]), adj], "x y - abs", func=mpeg2stinx).std.SeparateFields(tff)
        diff = MeanMode.MINIMUM([diff.std.SelectEvery(4, (0, 1)), diff.std.SelectEvery(4, (2, 3))], func=mpeg2stinx)
        diff = weave(Morpho.expand(diff, sw=2, sh=1), tff, mpeg2stinx)

        return norm_expr([flt, src, diff], "z {limit} * LIM! x y LIM@ - y LIM@ + clip", limit=limit, func=mpeg2stinx)

    def _bobfunc(clip: vs.VideoNode) -> vs.VideoNode:
        bobbed = bobber.bob(clip)

        if mask:
            bobbed = BWDIF(tff=tff, edeint=bobbed).bob(clip)

        return bobbed

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
