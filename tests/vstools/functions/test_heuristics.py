from vstools import core, video_heuristics, vs

clip = core.std.BlankClip(None, 640, 360, vs.YUV420P16).std.SetFrameProps(
    _Matrix=vs.MATRIX_BT709, _ColorRange=vs.RANGE_FULL
)


def test_video_heuristics_props_none() -> None:
    heuristics, assumed = video_heuristics(clip, None, assumed_return=True)

    assert heuristics == {"matrix_in": 6, "primaries_in": 6, "transfer_in": 6, "range_in": 1, "chromaloc_in": 0}
    assert assumed == ["_Matrix", "_Primaries", "_Transfer", "_ColorRange", "_ChromaLocation"]


def test_video_heuristics_props_false() -> None:
    heuristics, assumed = video_heuristics(clip, False, assumed_return=True)

    assert heuristics == {"matrix_in": 6, "primaries_in": 6, "transfer_in": 6, "range_in": 1, "chromaloc_in": 0}
    assert assumed == ["_Matrix", "_Primaries", "_Transfer", "_ColorRange", "_ChromaLocation"]


def test_video_heuristics_props_true() -> None:
    heuristics, assumed = video_heuristics(clip, True, assumed_return=True)

    assert heuristics == {"matrix_in": 1, "primaries_in": 6, "transfer_in": 6, "range_in": 0, "chromaloc_in": 0}
    assert assumed == ["_Primaries", "_Transfer", "_ChromaLocation"]


def test_video_heuristics_props_frameprops() -> None:
    with clip.get_frame(0) as f:
        props = f.props.copy()
    props.update(_ChromaLocation=2)

    heuristics, assumed = video_heuristics(clip, props, assumed_return=True)

    assert heuristics == {"matrix_in": 1, "primaries_in": 6, "transfer_in": 6, "range_in": 0, "chromaloc_in": 2}
    assert assumed == ["_Primaries", "_Transfer"]
