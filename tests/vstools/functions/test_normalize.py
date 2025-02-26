from unittest import TestCase

from vstools import flatten, normalize_ranges, vs


class TestNormalize(TestCase):
    def test_flatten(self) -> None:
        result: list[str] = flatten(["a", "b", ["c", "d", ["e"]]])  # type: ignore
        self.assertEqual(list(result), ["a", "b", "c", "d", "e"])

    def test_normalize_ranges(self) -> None:
        clip = vs.core.std.BlankClip(length=1000)

        self.assertEqual(normalize_ranges(clip, (None, None)), [(0, 999)])
        self.assertEqual(normalize_ranges(clip, (24, -24)), [(24, 975)])
        self.assertEqual(normalize_ranges(clip, [(24, 100), (80, 150)]), [(24, 150)])
