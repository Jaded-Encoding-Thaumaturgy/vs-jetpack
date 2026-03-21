from unittest import TestCase

import pytest

from vstools import FramesLengthError, shift_clip, shift_clip_multi, vs


class TestClip(TestCase):
    def test_shift_clip(self) -> None:
        clip = vs.core.std.BlankClip(length=12)
        result = shift_clip(clip, 1)
        assert result.num_frames == 12

    def test_shift_clip_negative(self) -> None:
        clip = vs.core.std.BlankClip(length=12)
        result = shift_clip(clip, -1)
        assert result.num_frames == 12

    def test_shift_clip_errors_if_offset_too_long(self) -> None:
        clip = vs.core.std.BlankClip(length=12)
        with pytest.raises(FramesLengthError):
            shift_clip(clip, 12)

    def test_shift_clip_multi(self) -> None:
        clip = vs.core.std.BlankClip(length=12)
        results = shift_clip_multi(clip, (-3, 3))
        assert len(results) == 7
