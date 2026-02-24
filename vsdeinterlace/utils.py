"""
Utilities for field-based processing and telecine pattern generation.
"""

from __future__ import annotations

from jetpytools import FuncExcept

from vstools import Field, FieldBased, FieldBasedLike, FieldLike, core, vs

__all__ = ["dmetrics", "get_field_difference", "reinterlace", "reweave", "telecine_patterns", "weave"]


def telecine_patterns(
    clipa: vs.VideoNode, clipb: vs.VideoNode, length: int = 5, func: FuncExcept | None = None
) -> list[vs.VideoNode]:
    """
    Generate all possible telecine patterns by interleaving frames from two clips.

    Args:
        clipa: First input clip.
        clipb: Second input clip.
        length: Cycle length used for frame selection. Defaults to 5.
        func: Function returned for custom error handling. This should only be set by VS package developers.

    Returns:
        A list of interleaved clips, each representing a unique telecine pattern.
    """
    func = func or telecine_patterns

    a_select = [clipa.std.SelectEvery(length, i) for i in range(length)]
    b_select = [clipb.std.SelectEvery(length, i) for i in range(length)]

    return [core.std.Interleave([(b_select if i == j else a_select)[j] for j in range(length)]) for i in range(length)]


def get_field_difference(
    clip: vs.VideoNode, tff: FieldBasedLike | bool | None = None, func: FuncExcept | None = None
) -> vs.VideoNode:
    """
    Compute the difference between top and bottom fields in a clip.

    Args:
        clip: Input clip.
        tff: Field order (top-field-first). If None, inferred from the clip. Defaults to None.
        func: Function returned for custom error handling. This should only be set by VS package developers.

    Returns:
        A clip with a per-frame property "FieldDifference" indicating the absolute difference between fields.
    """
    func = func or get_field_difference

    tff = FieldBased.from_param_or_video(tff, clip, True, func).is_tff

    stats = clip.std.SeparateFields(tff).std.PlaneStats()

    return core.akarin.PropExpr(
        [clip, stats[::2], stats[1::2]], lambda: {"FieldDifference": "y.PlaneStatsAverage z.PlaneStatsAverage - abs"}
    )


def weave(clip: vs.VideoNode, tff: FieldLike | bool | None = None, func: FuncExcept | None = None) -> vs.VideoNode:
    """
    Recombine fields into frames using DoubleWeave.

    Args:
        clip: Input clip.
        tff: Field order (top-field-first). If None, inferred from the clip. Defaults to None.
        func: Function returned for custom error handling. This should only be set by VS package developers.

    Returns:
        A clip with fields woven back into full frames.
    """
    func = func or weave

    return clip.std.DoubleWeave(Field.from_param_or_video(tff, clip, True, func).value)[::2]


def reweave(
    clipa: vs.VideoNode, clipb: vs.VideoNode, tff: FieldLike | bool | None = None, func: FuncExcept | None = None
) -> vs.VideoNode:
    """
    Interleave two clips and weave them into full frames.

    Args:
        clipa: First input clip.
        clipb: Second input clip.
        tff: Field order (top-field-first). If None, inferred from the clip. Defaults to None.
        func: Function returned for custom error handling. This should only be set by VS package developers.

    Returns:
        A reweaved clip with fields combined into frames.
    """
    func = func or reweave

    return weave(core.std.Interleave([clipa, clipb]).std.SelectEvery(4, (0, 1, 3, 2)), tff, func)


def reinterlace(clip: vs.VideoNode, tff: FieldBasedLike | bool, func: FuncExcept | None = None) -> vs.VideoNode:
    """
    Reinterlace a progressive clip by separating and weaving fields.

    Args:
        clip: Input clip.
        tff: Field order (top-field-first).
        func: Function returned for custom error handling. This should only be set by VS package developers.

    Returns:
        A reinterlaced clip with fields woven back into interlaced frames.
    """
    func = func or reinterlace

    tff = FieldBased.from_param(tff, func).is_tff

    return weave(clip.std.SeparateFields(tff).std.SelectEvery(4, (0, 3)), tff, func)


def dmetrics(
    clip: vs.VideoNode,
    tff: FieldBasedLike | bool | None = None,
    chroma: bool = True,
    nt: int = 10,
    y: tuple[int, int] = (0, 0),
    clip2: vs.VideoNode | None = None,
    func: FuncExcept | None = None,
) -> vs.VideoNode:
    """
    Attaches the match metrics calculated by Telecide (decomb package) to frames as properties. Primarily intended for
    use with Wobbly.

    Args:
        clip: Input clip.
        tff: Field order (top-field-first). If None, inferred from the clip. Defaults to None.
        chroma: Determines whether chroma combing is included in the decision made during postprocessing as to whether a
            frame is combed or not. Defaults to True.
        nt: Defines the noise tolerance threshold. Defaults to 10.
        y: Define an exclusion band for the field matching. If y0 is not equal to y1 this feature is enabled. Rows in
            the image between lines y0 and y1 (inclusive) are excluded from consideration when the field matching is
            decided. This feature is typically used to ignore subtitling, which might otherwise throw off the matching.
            Defaults to (0, 0).
        clip2: Clip that dmetrics will copy the frame properties to. If `clip2` is used, dmetrics will perform all
            calculations based on `clip`, but will copy the calculated metrics to `clip2`. This can be used to work
            around dmetrics's video format limitations. Defaults to None.
        func: Function returned for custom error handling. This should only be set by VS package developers.

    Returns:
        Clip with metrics attached as frame properties.
    """
    func = func or dmetrics

    tff = FieldBased.from_param_or_video(tff, clip, True, func).is_tff

    if clip2 is None and clip.format.id is not vs.YUV420P8:
        clip2 = clip
        clip = clip.resize.Bilinear(format=vs.YUV420P8)

    metrics = core.dmetrics.DMetrics(clip, tff, chroma, nt, y[0], y[1])

    if clip2 is not None:
        metrics = core.std.CopyFrameProps(clip2, clip, ("MMetrics", "VMetrics"))

    return metrics
