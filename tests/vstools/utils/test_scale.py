import numpy as np

from vstools import Range, scale_delta, scale_mask, scale_value, vs


def test_scale_value_no_change() -> None:
    assert scale_value(0, 8, 8) == 0
    assert scale_value(24, 8, 8) == 24
    assert scale_value(64, 8, 8) == 64
    assert scale_value(255, 8, 8) == 255


def test_scale_value_to_10bit() -> None:
    assert scale_value(0, 8, 10) == 0
    assert scale_value(24, 8, 10) == 96
    assert scale_value(64, 8, 10) == 256
    assert scale_value(255, 8, 10) == 1020


def test_scale_value_from_10bit() -> None:
    assert scale_value(0, 10, 8) == 0
    assert scale_value(96, 10, 8) == 24
    assert scale_value(256, 10, 8) == 64
    assert scale_value(1020, 10, 8) == 255


def test_scale_value_to_float() -> None:
    assert scale_value(0, 8, vs.YUV444PS) == -0.0730593607305936
    assert scale_value(24, 8, vs.YUV444PS) == 0.0365296803652968
    assert scale_value(64, 8, vs.YUV444PS) == 0.2191780821917808
    assert scale_value(255, 8, vs.YUV444PS) == 1.091324200913242


def test_scale_value_from_float() -> None:
    assert scale_value(0, vs.YUV444PS, 8) == 16
    assert scale_value(0.1, vs.YUV444PS, 8) == 38
    assert scale_value(0.25, vs.YUV444PS, 8) == 71
    assert scale_value(1, vs.YUV444PS, 8) == 235


def test_scale_value_to_limited() -> None:
    assert scale_value(0, 8, 8, Range.FULL, Range.LIMITED) == 16
    assert scale_value(24, 8, 8, Range.FULL, Range.LIMITED) == 37
    assert scale_value(64, 8, 8, Range.FULL, Range.LIMITED) == 71
    assert scale_value(255, 8, 8, Range.FULL, Range.LIMITED) == 235


def test_scale_value_from_limited() -> None:
    assert scale_value(0, 8, 8, Range.LIMITED, Range.FULL) == 0
    assert scale_value(24, 8, 8, Range.LIMITED, Range.FULL) == 9
    assert scale_value(64, 8, 8, Range.LIMITED, Range.FULL) == 56
    assert scale_value(235, 8, 8, Range.LIMITED, Range.FULL) == 255


def test_scale_value_numpy() -> None:
    arr = np.array([0, 24, 64, 255], dtype=np.uint8)

    # Scale 8-bit uint8 array to 10-bit uint16 array
    res_10 = scale_value(arr, 8, 10)
    assert np.array_equal(res_10, np.array([0, 96, 256, 1020], dtype=np.uint16))

    # Scale back to 8-bit uint8 array
    res_8 = scale_value(res_10, 10, 8)
    assert np.array_equal(res_8, np.array([0, 24, 64, 255], dtype=np.uint8))


def test_scale_value_numpy_format() -> None:
    arr = np.array([0, 24, 64, 255], dtype=np.uint8)
    # Using numpy arrays as format specification
    arr_in = np.zeros((10, 10), dtype=np.uint8)
    arr_out = np.zeros((10, 10), dtype=np.uint16)
    res_format = scale_value(arr, arr_in, arr_out)
    assert np.array_equal(res_format, np.array([0, 6144, 16384, 65280], dtype=np.uint16))


def test_scale_mask_numpy() -> None:
    arr = np.array([0, 24, 64, 255], dtype=np.uint8)
    # Scale mask
    res_mask = scale_mask(arr, 8, 10)
    assert np.array_equal(res_mask, np.array([0, 96, 257, 1023], dtype=np.uint16))


def test_scale_delta_numpy() -> None:
    arr = np.array([0, 24, 64, 255], dtype=np.uint8)
    # Scale delta
    res_delta = scale_delta(arr, 8, 10)
    assert np.array_equal(res_delta, np.array([0, 96, 256, 1020], dtype=np.uint16))
