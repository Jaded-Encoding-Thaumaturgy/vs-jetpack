from unittest import TestCase

import pytest

from vstools import UnsupportedFieldBasedError, vs
from vstools.enums.generic import ChromaLocation, FieldBased


class TestChromaLocation(TestCase):
    def test_from_res_uhd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=3840, height=2160)
        result = ChromaLocation.from_res(clip)
        assert result == ChromaLocation.LEFT

    def test_from_res_hd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=1920, height=1080)
        result = ChromaLocation.from_res(clip)
        assert result == ChromaLocation.LEFT

    def test_from_res_sd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=640, height=480)
        result = ChromaLocation.from_res(clip)
        assert result == ChromaLocation.LEFT

    def test_from_video_uhd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=3840, height=2160)
        result = ChromaLocation.from_video(clip)
        assert result == ChromaLocation.LEFT

    def test_from_video_hd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=1920, height=1080)
        result = ChromaLocation.from_video(clip)
        assert result == ChromaLocation.LEFT

    def test_from_video_sd(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=640, height=480)
        result = ChromaLocation.from_video(clip)
        assert result == ChromaLocation.LEFT

    def test_get_offsets(self) -> None:
        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        off_left, off_top = ChromaLocation.LEFT.get_offsets(clip)
        assert off_left == -0.5
        assert off_top == 0.0

        off_left, off_top = ChromaLocation.CENTER.get_offsets(clip)
        assert off_left == 0.0
        assert off_top == 0.0

        off_left, off_top = ChromaLocation.TOP_LEFT.get_offsets(clip)
        assert off_left == -0.5
        assert off_top == -0.5

        off_left, off_top = ChromaLocation.TOP.get_offsets(clip)
        assert off_left == 0.0
        assert off_top == -0.5

        off_left, off_top = ChromaLocation.BOTTOM_LEFT.get_offsets(clip)
        assert off_left == -0.5
        assert off_top == 0.5

        off_left, off_top = ChromaLocation.BOTTOM.get_offsets(clip)
        assert off_left == 0.0
        assert off_top == 0.5


class TestFieldBased(TestCase):
    def test_from_res(self) -> None:
        clip = vs.core.std.BlankClip()
        result = FieldBased.from_res(clip)
        assert result == FieldBased.PROGRESSIVE

        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = FieldBased.from_res(clip)
        assert result == FieldBased.PROGRESSIVE

    def test_from_video(self) -> None:
        clip = vs.core.std.BlankClip()
        result = FieldBased.from_video(clip)
        assert result == FieldBased.PROGRESSIVE

        clip = vs.core.std.BlankClip(format=vs.YUV420P8)
        result = FieldBased.from_video(clip)
        assert result == FieldBased.PROGRESSIVE

    def test_is_inter(self) -> None:
        assert FieldBased.TFF.is_inter
        assert FieldBased.BFF.is_inter
        assert not FieldBased.PROGRESSIVE.is_inter

    def test_field(self) -> None:
        assert FieldBased.TFF.field == 1
        assert FieldBased.BFF.field == 0
        with pytest.raises(UnsupportedFieldBasedError):
            FieldBased.PROGRESSIVE.field

    def test_is_tff(self) -> None:
        assert FieldBased.TFF.is_tff
        assert not FieldBased.BFF.is_tff
        with pytest.raises(UnsupportedFieldBasedError):
            assert not FieldBased.PROGRESSIVE.is_tff

    def test_inverted(self) -> None:
        assert FieldBased.TFF.inverted_field == FieldBased.BFF
        assert FieldBased.BFF.inverted_field == FieldBased.TFF
        with pytest.raises(UnsupportedFieldBasedError):
            FieldBased.PROGRESSIVE.inverted_field
