import pytest

from vsexprtools import combine, expr_func, norm_expr
from vstools import PlanesT, core, vs

clip_int8 = core.std.BlankClip(None, 2, 2, vs.YUV420P8, length=1, color=[128, 64, 64], keep=True)
clip_fp32 = core.std.BlankClip(None, 2, 2, vs.YUV420PS, length=1, color=[0.5, 0.25, 0.25], keep=True)


@pytest.mark.parametrize("clip", [clip_int8, clip_fp32])
def test_expr_func(clip: vs.VideoNode) -> None:
    expr_func(clip, "x dup *")


@pytest.mark.parametrize("clip", [clip_int8, clip_fp32])
def test_combine(clip: vs.VideoNode) -> None:
    combine([clip, clip])


@pytest.mark.parametrize("clip", [clip_int8, clip_fp32])
@pytest.mark.parametrize("planes", [0, [1, 2], [2]])
@pytest.mark.parametrize("format", [None, vs.GRAYH])
@pytest.mark.parametrize("split_planes", [False, True])
def test_norm_expr(
    clip: vs.VideoNode,
    planes: PlanesT,
    format: int | None,
    split_planes: bool,
) -> None:
    norm_expr(clip, "x")
