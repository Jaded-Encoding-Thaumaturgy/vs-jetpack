from collections.abc import Iterator
from typing import Any

import pytest

from vsexprtools import combine, expr_func, norm_expr
from vstools import PlanesT, core, vs


def get_params() -> Iterator[dict[str, Any]]:
    yield {"format": vs.YUV420P8, "color": [128, 64, 64]}
    yield {"format": vs.YUV420PS, "color": [0.5, 0.25, 0.25]}


@pytest.fixture(params=get_params(), ids=str)
def clip(request: pytest.FixtureRequest) -> vs.VideoNode:
    return core.std.BlankClip(None, 2, 2, length=1, keep=True, **request.param)


def test_expr_func(clip: vs.VideoNode) -> None:
    expr_func(clip, "x dup *")


def test_combine(clip: vs.VideoNode) -> None:
    combine([clip, clip])


@pytest.mark.parametrize("planes", [0, [1, 2], [2]])
def test_norm_expr(clip: vs.VideoNode, planes: PlanesT) -> None:
    norm_expr(clip, "x", planes)
