from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, SupportsFloat

from vsexprtools import ExprOp
from vstools import (
    ConstantFormatVideoNode,
    DitherType,
    FrameRangeN,
    FrameRangesN,
    HoldsVideoFormatT,
    VideoFormatT,
    check_ref_clip,
    check_variable,
    check_variable_format,
    depth,
    get_video_format,
    join,
    limiter,
    normalize_ranges,
    normalize_seq,
    replace_ranges,
    split,
    vs,
)

from .morpho import Morpho
from .types import Coordinates

__all__ = ["range_mask", "strength_zones_mask"]


@limiter
def range_mask(clip: vs.VideoNode, rad: int = 2, radc: int = 0) -> ConstantFormatVideoNode:
    assert check_variable(clip, range_mask)

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
    format: int | VideoFormatT | HoldsVideoFormatT = vs.GRAYS,
    length: int | None = None,
) -> ConstantFormatVideoNode:
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

    cache_strength_clips = dict[float, ConstantFormatVideoNode]()
    strength_clips = [base_clip]
    indices = [(0, n) for n in range(base_clip.num_frames)]

    for i, z in enumerate(zones, 1):
        rng, strength = z
        rng = normalize_ranges(base_clip, rng)

        for s, e in rng:
            e += bool(not replace_ranges.exclusive)
            indices[s:e] = [(i, n) for n in range(s, e)]

        if isinstance(strength, vs.VideoNode):
            if TYPE_CHECKING:
                assert check_variable_format(strength, strength_zones_mask)

            check_ref_clip(base_clip, strength, strength_zones_mask)

            strength_clips.append(strength)
            continue

        strength = 0 if strength is None else float(strength)

        if strength not in cache_strength_clips:
            cache_strength_clips[strength] = vs.core.std.BlankClip(base_clip, color=strength, keep=True)

        strength_clips.append(cache_strength_clips[strength])

    cache_strength_clips.clear()

    return vs.core.std.FrameEval(base_clip, lambda n: strength_clips[indices[n][0]][indices[n][1]], clip_src=base_clip)
