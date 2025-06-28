from __future__ import annotations

from vstools import ConstantFormatVideoNode, FieldBased, FieldBasedT, check_variable, core, vs

__all__ = ["get_field_difference", "reinterlace", "reweave", "telecine_patterns"]


def telecine_patterns(clipa: vs.VideoNode, clipb: vs.VideoNode, length: int = 5) -> list[ConstantFormatVideoNode]:
    assert check_variable(clipa, telecine_patterns)
    assert check_variable(clipb, telecine_patterns)

    a_select = [clipa.std.SelectEvery(length, i) for i in range(length)]
    b_select = [clipb.std.SelectEvery(length, i) for i in range(length)]

    return [core.std.Interleave([(b_select if i == j else a_select)[j] for j in range(length)]) for i in range(length)]


def get_field_difference(clip: vs.VideoNode, tff: FieldBasedT | bool | None = None) -> ConstantFormatVideoNode:
    assert check_variable(clip, get_field_difference)

    tff = FieldBased.from_param_or_video(tff, clip, True, get_field_difference).is_tff

    stats = clip.std.SeparateFields(tff).std.PlaneStats()

    return core.akarin.PropExpr(
        [clip, stats[::2], stats[1::2]], lambda: {"FieldDifference": "y.PlaneStatsAverage z.PlaneStatsAverage - abs"}
    )


def reinterlace(clip: vs.VideoNode, tff: FieldBasedT | bool | None = None) -> ConstantFormatVideoNode:
    assert check_variable(clip, reinterlace)

    tff = FieldBased.from_param_or_video(tff, clip, True, reinterlace).is_tff

    return clip.std.SeparateFields(tff).std.SelectEvery(4, (0, 3)).std.DoubleWeave(tff)[::2]


def reweave(clipa: vs.VideoNode, clipb: vs.VideoNode, tff: FieldBasedT | bool | None = None) -> ConstantFormatVideoNode:
    assert check_variable(clipa, reweave)
    assert check_variable(clipb, reweave)

    tff = FieldBased.from_param_or_video(tff, clipa, True, reweave).is_tff

    return core.std.Interleave([clipa, clipb]).std.SelectEvery(4, (0, 1, 3, 2)).std.DoubleWeave(tff)[::2]
