import pytest
from jetpytools import CustomRuntimeError

from vsdenoise import MotionVectors, MVDirection
from vstools import core, vs


def test_motion_vectors_init() -> None:
    mv = MotionVectors()
    assert mv.blksize is None
    assert mv.overlap_div is None

    mv2 = MotionVectors(blksize=16, overlap_div=2)
    assert mv2.blksize == (16, 16)
    assert mv2.overlap_div == (2, 2)


def test_motion_vectors_get_set() -> None:
    mv = MotionVectors()

    mv.blksize = (32, 16)
    assert mv.blksize == (32, 16)

    mv.blksize = None
    assert mv.blksize is None

    mv.overlap_div = (4, 2)
    assert mv.overlap_div == (4, 2)

    mv.overlap_div = None
    assert mv.overlap_div is None


def test_motion_vectors_overlap_div_fallback() -> None:
    mv = MotionVectors()
    assert mv.overlap_div is None

    # If overlap_div is None and blksize is None, returns None
    mv.blksize = None
    assert mv.overlap_div is None

    # If overlap_div is None and blksize is set, it falls back to analysis_data
    # But since analysis_data is empty/not present, it will raise KeyError and return None
    mv.blksize = (16, 16)
    assert mv.overlap_div is None


def test_motion_vectors_clear() -> None:
    mv = MotionVectors(blksize=16, overlap_div=2)
    dummy_clip = core.std.BlankClip()
    mv.set_vector(dummy_clip, MVDirection.BACKWARD, 1)

    assert mv.blksize == (16, 16)
    assert len(mv[MVDirection.BACKWARD]) == 1

    mv.clear()
    assert mv.blksize is None
    assert mv.overlap_div is None  # type: ignore[unreachable]
    assert len(mv[MVDirection.BACKWARD]) == 0


def test_motion_vectors_get_set_vector() -> None:
    mv = MotionVectors()
    dummy_clip = core.std.BlankClip()

    mv.set_vector(dummy_clip, MVDirection.BACKWARD, 1)
    assert mv.get_vector(MVDirection.BACKWARD, 1) is dummy_clip

    with pytest.raises(CustomRuntimeError):
        mv.get_vector(MVDirection.FORWARD, 1)


def test_motion_vectors_get_vectors() -> None:
    mv = MotionVectors()
    dummy1 = core.std.BlankClip()
    dummy2 = core.std.BlankClip()

    mv.set_vector(dummy1, MVDirection.BACKWARD, 1)
    mv.set_vector(dummy2, MVDirection.FORWARD, 2)

    # get_vectors with delta
    b, f = mv.get_vectors(direction=MVDirection.BOTH, delta=[1, 2])
    assert b == [dummy1]
    assert f == [dummy2]

    # get_vectors with tr
    b, f = mv.get_vectors(direction=MVDirection.BOTH, tr=2)
    assert b == [dummy1]
    assert f == [dummy2]

    # get_vectors with default deltas
    b, f = mv.get_vectors(direction=MVDirection.BOTH)
    assert b == [dummy1]
    assert f == [dummy2]

    # Test single directions
    b, f = mv.get_vectors(direction=MVDirection.BACKWARD, delta=1)
    assert b == [dummy1]
    assert f == []

    b, f = mv.get_vectors(direction=MVDirection.FORWARD, delta=2)
    assert b == []
    assert f == [dummy2]


def test_motion_vectors_deltas_tr() -> None:
    mv = MotionVectors()
    assert mv.deltas == []
    assert mv.tr == 0

    dummy = core.std.BlankClip()
    mv.set_vector(dummy, MVDirection.BACKWARD, 1)
    mv.set_vector(dummy, MVDirection.FORWARD, 3)

    assert mv.deltas == [1, 3]
    assert mv.tr == 3


def test_motion_vectors_analysis_data() -> None:
    clip = core.std.BlankClip(format=vs.GRAY8, width=16, height=16, length=1)
    clip = clip.std.SetFrameProp("MVUtensilsAnalysisBlkSizeX", 16)
    clip = clip.std.SetFrameProp("MVUtensilsAnalysisBlkSizeY", 8)
    clip = clip.std.SetFrameProp("MVUtensilsAnalysisOverlapX", 4)
    clip = clip.std.SetFrameProp("MVUtensilsAnalysisOverlapY", 2)
    clip = clip.std.SetFrameProp("OtherProp", 42)

    mv = MotionVectors()
    mv.set_vector(clip, MVDirection.BACKWARD, 1)

    # Access analysis_data
    data = mv.analysis_data
    assert data["MVUtensilsAnalysisBlkSizeX"] == 16
    assert data["MVUtensilsAnalysisBlkSizeY"] == 8
    assert data["MVUtensilsAnalysisOverlapX"] == 4
    assert data["MVUtensilsAnalysisOverlapY"] == 2
    assert "OtherProp" not in data

    # Test fallback blksize and overlap_div via analysis_data when they are None
    assert mv.blksize == (16, 8)
    assert mv.overlap_div == (4, 4)  # 16 // 4 = 4, 8 // 2 = 4

    # Deleter of analysis_data
    del mv.analysis_data
    # Cache should be cleared (accessing it again will recalculate)
    assert mv.analysis_data["MVUtensilsAnalysisBlkSizeX"] == 16
