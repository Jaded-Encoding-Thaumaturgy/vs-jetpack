import numpy as np

from vstools import (
    Range,
    core,
    get_lowest_value,
    get_lowest_values,
    get_neutral_value,
    get_neutral_values,
    get_peak_value,
    get_peak_values,
    scale_delta,
    scale_mask,
    scale_value,
    vs,
)


def test_scale_value_no_change() -> None:
    assert scale_value(0, 8, 8) == 0
    assert scale_value(24, 8, 8) == 24
    assert scale_value(64, 8, 8) == 64
    assert scale_value(255, 8, 8) == 255
    assert scale_value(0.42, 32, 32) == 0.42


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


def test_scale_value_round_and_clamp() -> None:
    assert scale_value(128.75, 8, 8) == 129
    assert scale_value(12.4, 8, 8) == 12
    assert scale_value(300, 8, 8) == 255
    assert scale_value(-5, 8, 8) == 0


def test_scale_value_numpy() -> None:
    arr = np.array([0, 24, 64, 255], dtype=np.uint8)

    # Scale 8-bit uint8 array to 10-bit uint16 array
    res_10 = scale_value(arr, 8, 10)
    assert np.array_equal(res_10, np.array([0, 96, 256, 1020], dtype=np.uint16))

    # Scale back to 8-bit uint8 array
    res_8 = scale_value(res_10, 10, 8)
    assert np.array_equal(res_8, np.array([0, 24, 64, 255], dtype=np.uint8))


def test_scale_value_numpy_no_change() -> None:
    arr = np.array([0, 24, 64, 255], dtype=np.uint8)
    res = scale_value(arr, 8, 8)
    assert np.array_equal(res, arr)
    assert res.dtype == arr.dtype

    arr = np.array([4096, 32768, 50000], dtype=np.uint16)
    res = scale_value(arr, 16, 16)
    assert np.array_equal(res, arr)
    assert res.dtype == arr.dtype

    arr = np.array([0.124532213251, 0.98656564, -0.521212], dtype=np.float32)
    res = scale_value(arr, 32, 32)
    assert np.array_equal(res, arr)
    assert res.dtype == arr.dtype


def test_scale_value_numpy_round_and_clamp() -> None:
    arr = np.array([128.75, 12.4], dtype=np.float32)
    res = scale_value(arr, 8, 8)
    assert np.array_equal(res, np.array([129, 12], dtype=np.uint8))

    arr = np.array([-5, 300], dtype=np.int16)
    res = scale_value(arr, 8, 8)
    assert np.array_equal(res, np.array([0, 255], dtype=np.uint8))


def test_scale_value_numpy_format() -> None:
    arr = np.array([0, 24, 64, 255], dtype=np.uint8)
    # Using numpy arrays as format specification
    arr_in = np.zeros((10, 10), dtype=np.uint8)
    arr_out = np.zeros((10, 10), dtype=np.uint16)
    res_format = scale_value(arr, arr_in, arr_out)
    assert np.array_equal(res_format, np.array([0, 6144, 16384, 65280], dtype=np.uint16))


def test_scale_value_numpy_dtypes_and_float() -> None:
    arr = np.array([0, 24, 64, 255], dtype=np.uint8)

    res_f32 = scale_value(arr, 8, vs.YUV444PS)
    assert res_f32.dtype == np.float32

    res_f16 = scale_value(arr, 8, vs.YUV444PH)
    assert res_f16.dtype == np.float16

    res_u32 = scale_value(arr, 8, vs.YUV444P32)
    assert res_u32.dtype == np.uint32

    # Fast path for same format array rounding
    arr_float_same = np.array([0.4, 24.4, 64.4, 127.5, 255.4], dtype=np.float64)
    res_same = scale_value(arr_float_same, 8, 8)
    assert np.array_equal(res_same, np.array([0, 24, 64, 128, 255], dtype=np.uint8))


def test_scale_value_video_node_and_frame() -> None:
    clip_8 = core.std.BlankClip(format=vs.YUV420P8)
    clip_10 = core.std.BlankClip(format=vs.YUV420P10)
    frame_8 = clip_8.get_frame(0)

    assert scale_value(16, clip_8, clip_10) == 64
    assert scale_value(16, frame_8, clip_10) == 64
    assert scale_value(16, clip_8, 10) == 64


def test_scale_value_rgb() -> None:
    clip_rgb = core.std.BlankClip(format=vs.RGB24)

    assert scale_value(0, clip_rgb, vs.RGB24) == 0
    assert scale_value(128, vs.RGB24, vs.RGB30, chroma=True) == 514
    assert scale_value(128, 8, 10, chroma=True, family=vs.RGB, range_in=Range.FULL, range_out=Range.FULL) == 514


def test_scale_value_chroma_offsets() -> None:
    assert scale_value(128, 8, 10, chroma=True) == 512
    assert scale_value(512, 10, 8, chroma=True) == 128
    assert scale_value(128, 8, vs.YUV444PS, chroma=True) == 0.0
    assert scale_value(0.0, vs.YUV444PS, 8, chroma=True) == 128


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


def test_scale_delta_video_node() -> None:
    clip_8 = core.std.BlankClip(format=vs.YUV420P8)
    clip_10 = core.std.BlankClip(format=vs.YUV420P10)
    frame_8 = clip_8.get_frame(0)

    assert scale_delta(10, clip_8, 10) == 40
    assert scale_delta(40, 10, clip_8) == 10
    assert scale_delta(10, clip_8, clip_10) == 40
    assert scale_delta(10, frame_8, 10) == 40


def test_get_lowest_value() -> None:
    assert get_lowest_value(8) == 16
    assert get_lowest_value(8, range_in=Range.FULL) == 0
    assert get_lowest_value(vs.YUV444PS) == 0.0
    assert get_lowest_value(vs.YUV444PS, chroma=True) == -0.5

    clip_8 = core.std.BlankClip(format=vs.YUV420P8)
    clip_rgb = core.std.BlankClip(format=vs.RGB24)
    frame_8 = clip_8.get_frame(0)

    assert get_lowest_value(clip_8) == 16
    assert get_lowest_value(frame_8) == 16
    assert get_lowest_value(clip_rgb) == 0
    assert get_lowest_value(vs.RGB24) == 0
    assert get_lowest_value(8, family=vs.RGB) == 0


def test_get_lowest_values() -> None:
    assert get_lowest_values(8) == [16, 16, 16]
    assert get_lowest_values(8, mask=True) == [0, 0, 0]
    assert get_lowest_values(vs.YUV420P8) == [16, 16, 16]
    assert get_lowest_values(vs.YUV444PS) == [0.0, -0.5, -0.5]


def test_get_neutral_value() -> None:
    assert get_neutral_value(8) == 128
    assert get_neutral_value(10) == 512
    assert get_neutral_value(16) == 32768
    assert get_neutral_value(vs.YUV444PS) == 0.0


def test_get_neutral_values() -> None:
    assert get_neutral_values(8) == [128, 128, 128]
    assert get_neutral_values(vs.YUV444PS) == [0.0, 0.0, 0.0]
    clip_8 = core.std.BlankClip(format=vs.YUV420P8)
    assert get_neutral_values(clip_8) == [128, 128, 128]


def test_get_peak_value() -> None:
    assert get_peak_value(8) == 235
    assert get_peak_value(8, chroma=True) == 240
    assert get_peak_value(8, range_in=Range.FULL) == 255
    assert get_peak_value(vs.YUV444PS) == 1.0
    assert get_peak_value(vs.YUV444PS, chroma=True) == 0.5

    clip_8 = core.std.BlankClip(format=vs.YUV420P8)
    clip_rgb = core.std.BlankClip(format=vs.RGB24)
    frame_8 = clip_8.get_frame(0)

    assert get_peak_value(clip_8) == 235
    assert get_peak_value(frame_8) == 235
    assert get_peak_value(clip_rgb) == 255
    assert get_peak_value(vs.RGB24) == 255
    assert get_peak_value(8, family=vs.RGB) == 255


def test_get_peak_values() -> None:
    assert get_peak_values(8) == [235, 240, 240]
    assert get_peak_values(8, mask=True) == [255, 255, 255]
    assert get_peak_values(vs.YUV420P8) == [235, 240, 240]
    assert get_peak_values(vs.YUV444PS) == [1.0, 0.5, 0.5]
