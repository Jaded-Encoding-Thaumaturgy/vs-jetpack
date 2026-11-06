from vsdenoise import refine_blksize


def test_refine_blksize() -> None:
    assert refine_blksize(16) == (8, 8)
    assert refine_blksize((16, 8), (2, 2)) == (8, 4)
    assert refine_blksize((16, 16), (0, 0)) == (0, 0)
    assert refine_blksize(8, 2) == (4, 4)
