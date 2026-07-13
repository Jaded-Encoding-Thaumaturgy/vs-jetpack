import pytest

from vstools import core, video_heuristics, vs


@pytest.fixture
def clip() -> vs.VideoNode:
    return core.std.BlankClip(None, 640, 360, vs.YUV420P16).std.SetFrameProps(
        _Matrix=vs.MATRIX_BT709, _Range=vs.RANGE_FULL
    )


def test_video_heuristics_props_none(clip: vs.VideoNode) -> None:
    heuristics, assumed = video_heuristics(clip, None, assumed_return=True)

    assert heuristics == {
        "matrix_in": 6,
        "primaries_in": 6,
        "transfer_in": 6,
        "range_in": vs.RANGE_LIMITED,
        "chromaloc_in": 0,
    }
    assert assumed == ["_Matrix", "_Primaries", "_Transfer", "_Range", "_ChromaLocation"]


def test_video_heuristics_props_false(clip: vs.VideoNode) -> None:
    heuristics, assumed = video_heuristics(clip, False, assumed_return=True)

    assert heuristics == {
        "matrix_in": 6,
        "primaries_in": 6,
        "transfer_in": 6,
        "range_in": vs.RANGE_LIMITED,
        "chromaloc_in": 0,
    }
    assert assumed == ["_Matrix", "_Primaries", "_Transfer", "_Range", "_ChromaLocation"]


@pytest.mark.filterwarnings("ignore:The _ColorRange frame property has been deprecated, use _Range instead")
def test_video_heuristics_props_true(clip: vs.VideoNode) -> None:
    heuristics, assumed = video_heuristics(clip, True, assumed_return=True)

    assert heuristics == {
        "matrix_in": 1,
        "primaries_in": 6,
        "transfer_in": 6,
        "range_in": vs.RANGE_FULL,
        "chromaloc_in": 0,
    }
    assert assumed == ["_Primaries", "_Transfer", "_ChromaLocation"]


@pytest.mark.filterwarnings("ignore:The _ColorRange frame property has been deprecated, use _Range instead")
def test_video_heuristics_props_frameprops(clip: vs.VideoNode) -> None:
    with clip.get_frame(0) as f:
        props = f.props.copy()
    props.update(_ChromaLocation=2)

    heuristics, assumed = video_heuristics(clip, props, assumed_return=True)

    assert heuristics == {
        "matrix_in": 1,
        "primaries_in": 6,
        "transfer_in": 6,
        "range_in": vs.RANGE_FULL,
        "chromaloc_in": 2,
    }
    assert assumed == ["_Primaries", "_Transfer"]
