from unittest import TestCase

from vstools import FramesLengthError, shift_clip, shift_clip_multi, vs


class TestClip(TestCase):
    def test_shift_clip(self) -> None:
        clip = vs.core.std.BlankClip(length=12)
        result = shift_clip(clip, 1)
        self.assertEqual(result.num_frames, 12)

    def test_shift_clip_negative(self) -> None:
        clip = vs.core.std.BlankClip(length=12)
        result = shift_clip(clip, -1)
        self.assertEqual(result.num_frames, 12)

    def test_shift_clip_errors_if_offset_too_long(self) -> None:
        clip = vs.core.std.BlankClip(length=12)
        with self.assertRaises(FramesLengthError):
            shift_clip(clip, 12)

    def test_shift_clip_multi(self) -> None:
        clip = vs.core.std.BlankClip(length=12)
        results = shift_clip_multi(clip, (-3, 3))
        self.assertEqual(len(results), 7)
