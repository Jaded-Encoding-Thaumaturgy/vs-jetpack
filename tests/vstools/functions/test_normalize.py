from unittest import TestCase

from vstools import flatten, invert_ranges, normalize_ranges, vs


class TestNormalize(TestCase):
    def test_flatten(self) -> None:
        result: list[str] = flatten(["a", "b", ["c", "d", ["e"]]])  # type: ignore
        assert list(result) == ["a", "b", "c", "d", "e"]

    def test_normalize_ranges(self) -> None:
        clip = vs.core.std.BlankClip(length=1000)

        # exclusive=False
        assert normalize_ranges(clip, (None, None)) == [(0, 999)]
        assert normalize_ranges(clip, (24, -24)) == [(24, 975)]
        assert normalize_ranges(clip, [(24, 100), (80, 150)]) == [(24, 150)]
        assert normalize_ranges(clip, [100, 500]) == [(100, 100), (500, 500)]
        assert normalize_ranges(clip, (-100, 950)) == [(900, 950)]

        # exclusive=True
        assert normalize_ranges(clip, (None, None), True) == [(0, 1000)]
        assert normalize_ranges(clip, (24, -24), True) == [(24, 976)]
        assert normalize_ranges(clip, [(24, 100), (80, 150)], True) == [(24, 150)]
        assert normalize_ranges(clip, [100, 500], True) == [(100, 101), (500, 501)]
        assert normalize_ranges(clip, (-100, 950), True) == [(900, 950)]

    def test_invert_ranges_ranges(self) -> None:
        clip_a = vs.core.std.BlankClip(length=1000)
        # clip_b = vs.core.std.BlankClip(length=500)

        # exclusive=False
        assert invert_ranges(clip_a, None, (None, None)) == []
        assert invert_ranges(clip_a, None, (24, -24)) == [(0, 23), (976, 999)]
        assert invert_ranges(clip_a, None, [100, 500]) == [(0, 99), (101, 499), (501, 999)]
        assert invert_ranges(clip_a, None, [(24, 100), (80, 150)]) == [(0, 23), (151, 999)]

        assert invert_ranges(clip_a, None, (-100, 950)) == [(0, 899), (951, 999)]

        # exclusive=True
        assert invert_ranges(clip_a, None, (None, None), True) == []
        assert invert_ranges(clip_a, None, (24, -24), True) == [(0, 24), (976, 1000)]
        assert invert_ranges(clip_a, None, [100, 500], True) == [(0, 100), (101, 500), (501, 1000)]
        assert invert_ranges(clip_a, None, [(24, 100), (80, 150)], True) == [(0, 24), (150, 1000)]

        assert invert_ranges(clip_a, None, (-100, 950), True) == [(0, 900), (950, 1000)]
