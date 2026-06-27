from unittest import TestCase

import numpy as np

from vstools import Range, scale_delta, scale_mask, scale_value, vs


class TestScale(TestCase):
    def test_scale_value_no_change(self) -> None:
        result = scale_value(0, 8, 8)
        self.assertEqual(result, 0)

        result = scale_value(24, 8, 8)
        self.assertEqual(result, 24)

        result = scale_value(64, 8, 8)
        self.assertEqual(result, 64)

        result = scale_value(255, 8, 8)
        self.assertEqual(result, 255)

    def test_scale_value_to_10bit(self) -> None:
        result = scale_value(0, 8, 10)
        self.assertEqual(result, 0)

        result = scale_value(24, 8, 10)
        self.assertEqual(result, 96)

        result = scale_value(64, 8, 10)
        self.assertEqual(result, 256)

        result = scale_value(255, 8, 10)
        self.assertEqual(result, 1020)

    def test_scale_value_from_10bit(self) -> None:
        result = scale_value(0, 10, 8)
        self.assertEqual(result, 0)

        result = scale_value(96, 10, 8)
        self.assertEqual(result, 24)

        result = scale_value(256, 10, 8)
        self.assertEqual(result, 64)

        result = scale_value(1020, 10, 8)
        self.assertEqual(result, 255)

    def test_scale_value_to_float(self) -> None:
        result = scale_value(0, 8, vs.YUV444PS)
        self.assertEqual(result, -0.0730593607305936)

        result = scale_value(24, 8, vs.YUV444PS)
        self.assertEqual(result, 0.0365296803652968)

        result = scale_value(64, 8, vs.YUV444PS)
        self.assertEqual(result, 0.2191780821917808)

        result = scale_value(255, 8, vs.YUV444PS)
        self.assertEqual(result, 1.091324200913242)

    def test_scale_value_from_float(self) -> None:
        result = scale_value(0, vs.YUV444PS, 8)
        self.assertEqual(result, 16)

        result = scale_value(0.1, vs.YUV444PS, 8)
        self.assertEqual(result, 38)

        result = scale_value(0.25, vs.YUV444PS, 8)
        self.assertEqual(result, 71)

        result = scale_value(1, vs.YUV444PS, 8)
        self.assertEqual(result, 235)

    def test_scale_value_to_limited(self) -> None:
        result = scale_value(0, 8, 8, Range.FULL, Range.LIMITED)
        self.assertEqual(result, 16)

        result = scale_value(24, 8, 8, Range.FULL, Range.LIMITED)
        self.assertEqual(result, 37)

        result = scale_value(64, 8, 8, Range.FULL, Range.LIMITED)
        self.assertEqual(result, 71)

        result = scale_value(255, 8, 8, Range.FULL, Range.LIMITED)
        self.assertEqual(result, 235)

    def test_scale_value_from_limited(self) -> None:
        result = scale_value(0, 8, 8, Range.LIMITED, Range.FULL)
        self.assertEqual(result, 0)

        result = scale_value(24, 8, 8, Range.LIMITED, Range.FULL)
        self.assertEqual(result, 9)

        result = scale_value(64, 8, 8, Range.LIMITED, Range.FULL)
        self.assertEqual(result, 56)

        result = scale_value(235, 8, 8, Range.LIMITED, Range.FULL)
        self.assertEqual(result, 255)

    def test_scale_value_numpy(self) -> None:
        arr = np.array([0, 24, 64, 255], dtype=np.uint8)

        # Scale 8-bit uint8 array to 10-bit uint16 array
        res_10 = scale_value(arr, 8, 10)
        self.assertTrue(np.array_equal(res_10, np.array([0, 96, 256, 1020], dtype=np.uint16)))

        # Scale back to 8-bit uint8 array
        res_8 = scale_value(res_10, 10, 8)
        self.assertTrue(np.array_equal(res_8, np.array([0, 24, 64, 255], dtype=np.uint8)))

        # Using numpy arrays as format specification
        arr_in = np.zeros((10, 10), dtype=np.uint8)
        arr_out = np.zeros((10, 10), dtype=np.uint16)
        res_format = scale_value(arr, arr_in, arr_out)
        self.assertTrue(np.array_equal(res_format, np.array([0, 6144, 16384, 65280], dtype=np.uint16)))

        # Scale mask
        res_mask = scale_mask(arr, 8, 10)
        self.assertTrue(np.array_equal(res_mask, np.array([0, 96, 257, 1023], dtype=np.uint16)))

        # Scale delta
        res_delta = scale_delta(arr, 8, 10)
        self.assertTrue(np.array_equal(res_delta, np.array([0, 96, 256, 1020], dtype=np.uint16)))
