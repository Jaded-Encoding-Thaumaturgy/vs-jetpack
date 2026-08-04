import numpy as np

from vstools import get_depth, get_h, get_sample_type, get_w, vs


def test_get_w() -> None:
    assert get_w(1080, 16 / 9) == 1920
    assert get_w(1080, 4 / 3) == 1440

    clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=1920, height=1080)
    assert get_w(1080, clip) == 1920


def test_get_h() -> None:
    assert get_h(1920, 16 / 9) == 1080
    assert get_h(1440, 4 / 3) == 1080

    clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=1920, height=1080)
    assert get_h(1920, clip) == 1080


def test_get_video_format_numpy_dtypes() -> None:
    # Dtypes
    assert get_depth(np.uint8) == 8
    assert get_sample_type(np.uint8) == vs.INTEGER

    assert get_depth(np.dtype("uint16")) == 16
    assert get_sample_type(np.dtype("uint16")) == vs.INTEGER

    assert get_depth(np.float32) == 32
    assert get_sample_type(np.float32) == vs.FLOAT


def test_get_video_format_numpy_arrays() -> None:
    # Arrays
    arr_u8 = np.zeros((10, 10), dtype=np.uint8)
    assert get_depth(arr_u8) == 8
    assert get_sample_type(arr_u8) == vs.INTEGER

    arr_f32 = np.zeros((10, 10), dtype=np.float32)
    assert get_depth(arr_f32) == 32
    assert get_sample_type(arr_f32) == vs.FLOAT


def test_get_video_format_numpy_strings() -> None:
    # String representations
    assert get_depth("uint8") == 8
    assert get_sample_type("uint8") == vs.INTEGER

    assert get_depth("float32") == 32
    assert get_sample_type("float32") == vs.FLOAT
