from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import SupportsFloat

from jetpytools import FuncExcept, clamp, normalize_seq

from vsexprtools import ExprOp
from vsrgtools import BlurMatrix, median_blur
from vstools import (
    ConvMode,
    DitherType,
    FrameRangeN,
    FrameRangesN,
    HoldsVideoFormat,
    Planes,
    VideoFormatLike,
    check_ref_clip,
    core,
    depth,
    get_video_format,
    join,
    limiter,
    normalize_ranges,
    replace_ranges,
    scale_mask,
    split,
    vs,
)

from .morpho import Morpho
from .types import Coordinates

__all__ = ["range_mask", "stabilize_mask", "strength_zones_mask"]


@limiter(mask=True)
def range_mask(clip: vs.VideoNode, rad: int = 2, radc: int = 0) -> vs.VideoNode:
    def _minmax(clip: vs.VideoNode, iters: int, maxx: bool) -> vs.VideoNode:
        func = Morpho.maximum if maxx else Morpho.minimum

        for i in range(1, iters + 1):
            clip = func(clip, coordinates=Coordinates.from_iter(i))

        return clip

    return join(
        [
            ExprOp.SUB.combine(_minmax(plane, r, True), _minmax(plane, r, False))
            for plane, r in zip(split(clip), normalize_seq((radc and [rad, radc]) or rad, clip.format.num_planes))
        ]
    )


def strength_zones_mask(
    base: SupportsFloat | vs.VideoNode | None = None,
    zones: Sequence[tuple[FrameRangeN | FrameRangesN, SupportsFloat | vs.VideoNode | None]] | None = None,
    format: int | VideoFormatLike | HoldsVideoFormat = vs.GRAYS,
    length: int | None = None,
) -> vs.VideoNode:
    """
    Creates a mask based on a threshold strength, with optional adjustments using defined zones.

    Args:
        base: The base clip used to generate the strength mask. If set to None, a blank mask (all zeros) will be created
            using the specified format.
        zones: Optional list of zones to define varying strength regions. Defaults to None.
        format: Pixel format for the mask. Defaults to vs.GRAYS.
        length: Total number of frames for the mask. If None, uses the length of the base clip.

    Returns:
        A mask clip representing the defined strength zones.
    """

    if isinstance(base, vs.VideoNode):
        base_clip = depth(base, format, dither_type=DitherType.NONE)

        if length:
            if base_clip.num_frames > length:
                base_clip = base_clip[:length]

            if base_clip.num_frames < length:
                base_clip = base_clip + base_clip[-1] * (length - base_clip.num_frames)

    elif base is None:
        base_clip = vs.core.std.BlankClip(format=get_video_format(format).id, length=length, color=base, keep=True)
    else:
        base_clip = vs.core.std.BlankClip(
            format=get_video_format(format).id, length=length, color=float(base), keep=True
        )

    if not zones:
        return base_clip

    cache_strength_clips = dict[float, vs.VideoNode]()
    strength_clips = [base_clip]
    indices = [(0, n) for n in range(base_clip.num_frames)]

    for i, z in enumerate(zones, 1):
        rng, strength = z
        rng = normalize_ranges(base_clip, rng)

        for s, e in rng:
            e += bool(not replace_ranges.exclusive)
            indices[s:e] = [(i, n) for n in range(s, e)]

        if isinstance(strength, vs.VideoNode):
            check_ref_clip(base_clip, strength, strength_zones_mask)
            strength_clips.append(strength)
            continue

        strength = 0 if strength is None else float(strength)

        if strength not in cache_strength_clips:
            cache_strength_clips[strength] = vs.core.std.BlankClip(base_clip, color=strength, keep=True)

        strength_clips.append(cache_strength_clips[strength])

    cache_strength_clips.clear()

    return vs.core.std.FrameEval(base_clip, lambda n: strength_clips[indices[n][0]][indices[n][1]], clip_src=base_clip)


def stabilize_mask(
    clip: vs.VideoNode,
    radius: int = 3,
    ranges: FrameRangeN | FrameRangesN = None,
    scenechanges: Iterable[int] | None = None,
    kernel: BlurMatrix = BlurMatrix.MEAN_NO_CENTER,
    brz: int = 0,
    planes: Planes = None,
    func: FuncExcept | None = None,
) -> vs.VideoNode:
    """
    Generate a stabilization mask highlighting unstable regions between frames using temporal median and blur filtering.

    Useful for stabilizing credit masks.

    Args:
        clip: Input mask.
        radius: Temporal radius for filtering. Higher values smooth more. Defaults to 3.
        ranges: Frame ranges treated as scene changes.
        scenechanges: Explicit list of scenechange frames.
        kernel: Blur kernel applied after the median step.
        brz: Threshold bias for binarization. Higher = stricter. Defaults to 0.
        planes: Which planes to process. Defaults to all.
        func: Function returned for custom error handling. This should only be set by VS package developers.

    Returns:
        A mask clip where white marks unstable areas and black marks stable regions.
    """
    func = func or stabilize_mask

    frames = list[int]()
    scprev, scnext = set[int](), set[int]()

    if ranges:
        frames.extend(x for s, e in normalize_ranges(clip, ranges) for x in (s, e + int(not replace_ranges.exclusive)))

    if scenechanges:
        frames.extend(scenechanges)

    for f in frames:
        scprev.add(f)
        scnext.add(f - 1)

    def set_scenechanges(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        f = f.copy()
        f.props.update(
            _SceneChangePrev=f.props.get("_SceneChangePrev", False),
            _SceneChangeNext=f.props.get("_SceneChangeNext", False),
        )

        if n in scprev:
            f.props["_SceneChangePrev"] = True

        if n in scnext:
            f.props["_SceneChangeNext"] = True

        return f

    if scprev or scnext:
        clip = core.std.ModifyFrame(clip, clip, set_scenechanges)

    median = median_blur(clip, radius, ConvMode.TEMPORAL, planes, scenechange=True, func=func)

    radius_blur = radius * 2 + 1
    blurred = kernel(radius_blur, mode=ConvMode.TEMPORAL)(median, planes, scenechange=True, func=func)

    binarized = blurred.std.Binarize(
        scale_mask(1.0 / (radius_blur - clamp(brz, 0, radius_blur)), 32, clip), planes=planes
    )

    return binarized
