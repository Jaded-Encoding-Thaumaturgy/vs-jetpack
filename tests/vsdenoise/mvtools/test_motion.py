from vsdenoise import MotionVectors, MVDirection
from vstools import core


def test_motion_vectors_init() -> None:
    mv = MotionVectors()
    assert mv.blksize is None
    assert mv.overlap is None

    mv2 = MotionVectors(blksize=(16, 16), overlap=(8, 8))
    assert mv2.blksize == (16, 16)
    assert mv2.overlap == (8, 8)


def test_motion_vectors_get_set() -> None:
    mv = MotionVectors()

    mv.blksize = (32, 16)
    assert mv.blksize == (32, 16)

    mv.blksize = None
    assert mv.blksize is None

    mv.overlap = (8, 8)
    assert mv.overlap == (8, 8)

    mv.overlap = None
    assert mv.overlap is None


def test_motion_vectors_overlap_div_fallback() -> None:
    mv = MotionVectors()
    assert mv.overlap is None

    # If overlap_div is None and blksize is None, returns None
    mv.blksize = None
    assert mv.overlap is None

    # If overlap_div is None and blksize is set, it falls back to analysis_data
    # But since analysis_data is empty/not present, it will raise KeyError and return None
    mv.blksize = (16, 16)
    assert mv.overlap is None


def test_motion_vectors_clear() -> None:
    mv = MotionVectors(blksize=(16, 16), overlap=(8, 8))

    dummy_clip = core.std.BlankClip()
    mv[1] = dummy_clip

    assert mv.blksize == (16, 16)
    assert len(mv) == 1

    mv.clear()
    assert mv.blksize is None
    assert mv.overlap is None  # type: ignore[unreachable]
    assert len(mv) == 0


def test_motion_vectors_get_vectors() -> None:
    mv = MotionVectors()
    dummy1 = core.std.BlankClip()
    dummy2 = core.std.BlankClip()

    mv[1] = dummy1
    mv[-1] = dummy2

    # get_vectors with delta
    b, f = mv.get_vectors(delta=[-1, 1])
    assert b == [dummy1]
    assert f == [dummy2]

    # get_vectors with tr
    b, f = mv.get_vectors(direction=MVDirection.BOTH, tr=1)
    assert b == [dummy1]
    assert f == [dummy2]

    # Test single directions
    b, f = mv.get_vectors(direction=MVDirection.BACKWARD, tr=1)
    assert b == [dummy1]
    assert f == []

    b, f = mv.get_vectors(direction=MVDirection.FORWARD, tr=1)
    assert b == []
    assert f == [dummy2]
