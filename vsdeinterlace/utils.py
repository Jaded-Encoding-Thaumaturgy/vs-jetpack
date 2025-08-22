from __future__ import annotations

from vstools import ConstantFormatVideoNode, FieldBased, FieldBasedLike, check_variable, core, vs

__all__ = ["get_field_difference", "reinterlace", "reweave", "telecine_patterns", "weave"]


def telecine_patterns(clipa: vs.VideoNode, clipb: vs.VideoNode, length: int = 5) -> list[ConstantFormatVideoNode]:
    assert check_variable(clipa, telecine_patterns)
    assert check_variable(clipb, telecine_patterns)

    a_select = [clipa.std.SelectEvery(length, i) for i in range(length)]
    b_select = [clipb.std.SelectEvery(length, i) for i in range(length)]

    return [core.std.Interleave([(b_select if i == j else a_select)[j] for j in range(length)]) for i in range(length)]


def get_field_difference(clip: vs.VideoNode, tff: FieldBasedLike | bool | None = None) -> ConstantFormatVideoNode:
    assert check_variable(clip, get_field_difference)

    tff = FieldBased.from_param_or_video(tff, clip, True, get_field_difference).is_tff

    stats = clip.std.SeparateFields(tff).std.PlaneStats()

    return core.akarin.PropExpr(
        [clip, stats[::2], stats[1::2]], lambda: {"FieldDifference": "y.PlaneStatsAverage z.PlaneStatsAverage - abs"}
    )


def weave(clip: vs.VideoNode, tff: FieldBasedLike | bool | None = None) -> ConstantFormatVideoNode:
    assert check_variable(clip, weave)

    tff = FieldBased.from_param_or_video(tff, clip, True, weave).is_tff

    return clip.std.DoubleWeave(tff)[::2]


def reweave(
    clipa: vs.VideoNode, clipb: vs.VideoNode, tff: FieldBasedLike | bool | None = None
) -> ConstantFormatVideoNode:
    assert check_variable(clipa, reweave)
    assert check_variable(clipb, reweave)

    return weave(core.std.Interleave([clipa, clipb]), tff)


def reinterlace(clip: vs.VideoNode, tff: FieldBasedLike | bool | None = None) -> ConstantFormatVideoNode:
    assert check_variable(clip, reinterlace)

    tff = FieldBased.from_param_or_video(tff, clip, True, reinterlace).is_tff

    return weave(clip.std.SeparateFields(tff).std.SelectEvery(4, (0, 3)), tff)
