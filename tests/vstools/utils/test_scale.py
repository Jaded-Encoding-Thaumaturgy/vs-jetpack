from unittest import TestCase

from vstools import ColorRange, scale_value, vs


class TestScale(TestCase):
    def test_scale_value_no_change(self) -> None:
        result = scale_value(0, 8, 8)
        assert result == 0

        result = scale_value(24, 8, 8)
        assert result == 24

        result = scale_value(64, 8, 8)
        assert result == 64

        result = scale_value(255, 8, 8)
        assert result == 255

    def test_scale_value_to_10bit(self) -> None:
        result = scale_value(0, 8, 10)
        assert result == 0

        result = scale_value(24, 8, 10)
        assert result == 96

        result = scale_value(64, 8, 10)
        assert result == 256

        result = scale_value(255, 8, 10)
        assert result == 1020

    def test_scale_value_from_10bit(self) -> None:
        result = scale_value(0, 10, 8)
        assert result == 0

        result = scale_value(96, 10, 8)
        assert result == 24

        result = scale_value(256, 10, 8)
        assert result == 64

        result = scale_value(1020, 10, 8)
        assert result == 255

    def test_scale_value_to_float(self) -> None:
        result = scale_value(0, 8, vs.YUV444PS)
        assert result == -0.0730593607305936

        result = scale_value(24, 8, vs.YUV444PS)
        assert result == 0.0365296803652968

        result = scale_value(64, 8, vs.YUV444PS)
        assert result == 0.2191780821917808

        result = scale_value(255, 8, vs.YUV444PS)
        assert result == 1.091324200913242

    def test_scale_value_from_float(self) -> None:
        result = scale_value(0, vs.YUV444PS, 8)
        assert result == 16

        result = scale_value(0.1, vs.YUV444PS, 8)
        assert result == 38

        result = scale_value(0.25, vs.YUV444PS, 8)
        assert result == 71

        result = scale_value(1, vs.YUV444PS, 8)
        assert result == 235

    def test_scale_value_to_limited(self) -> None:
        result = scale_value(0, 8, 8, ColorRange.FULL, ColorRange.LIMITED)
        assert result == 16

        result = scale_value(24, 8, 8, ColorRange.FULL, ColorRange.LIMITED)
        assert result == 37

        result = scale_value(64, 8, 8, ColorRange.FULL, ColorRange.LIMITED)
        assert result == 71

        result = scale_value(255, 8, 8, ColorRange.FULL, ColorRange.LIMITED)
        assert result == 235

    def test_scale_value_from_limited(self) -> None:
        result = scale_value(0, 8, 8, ColorRange.LIMITED, ColorRange.FULL)
        assert result == 0

        result = scale_value(24, 8, 8, ColorRange.LIMITED, ColorRange.FULL)
        assert result == 9

        result = scale_value(64, 8, 8, ColorRange.LIMITED, ColorRange.FULL)
        assert result == 56

        result = scale_value(235, 8, 8, ColorRange.LIMITED, ColorRange.FULL)
        assert result == 255
