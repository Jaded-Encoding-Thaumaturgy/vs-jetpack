from vsdenoise import normalize_thscd, refine_blksize


def test_refine_blksize() -> None:
    assert refine_blksize(16) == (8, 8)
    assert refine_blksize((16, 8), (2, 2)) == (8, 4)
    assert refine_blksize((16, 16), (0, 0)) == (0, 0)
    assert refine_blksize(8, 2) == (4, 4)


def test_normalize_thscd() -> None:
    assert normalize_thscd(None) == (None, None)
    assert normalize_thscd(400) == (400, None)
    assert normalize_thscd((400, 0.5)) == (400, 0.5)
