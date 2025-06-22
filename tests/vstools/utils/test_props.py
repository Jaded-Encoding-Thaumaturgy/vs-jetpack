from typing import Any, Literal

import pytest

from vstools import FramePropError, core, get_prop, merge_clip_props, vs

clip = core.std.BlankClip(
    format=vs.YUV420P8,
    width=1920,
    height=1080,
)

clip = core.std.SetFrameProps(
    clip,
    _Matrix=1,
    _Transfer=1,
    _Primaries=1,
    __StrProp="test string",
    __IntProp=123,
    __FloatProp=123.456,
    __BoolProp=True,
    __BytesProp=b"test bytes",
    __VideoFrameProp=clip.get_frame(0),
)

video_frame = clip.get_frame(0)

clip2 = core.std.BlankClip(
    format=vs.YUV420P8,
    width=1920,
    height=1080,
)

clip2 = core.std.SetFrameProps(clip2, _Matrix=5, _RandomProp=1, __AnotherRandomProp="gsdgsdgs")


@pytest.mark.parametrize("prop_name", ["_Matrix", "_Transfer", "_Primaries"])
def test_get_prop_video_node_input(prop_name: str) -> None:
    """Test get_prop with VideoNode input."""
    assert get_prop(clip, prop_name, int) == 1


@pytest.mark.parametrize("prop_name", ["_Matrix", "_Transfer", "_Primaries"])
def test_get_prop_video_frame_input(prop_name: str) -> None:
    """Test get_prop with VideoFrame input."""
    assert get_prop(video_frame, prop_name, int) == 1


@pytest.mark.parametrize("prop_name", ["_Matrix", "_Transfer", "_Primaries"])
def test_get_prop_frame_props_input(prop_name: str) -> None:
    """Test get_prop with FrameProps input."""
    assert get_prop(video_frame.props, prop_name, int) == 1


def test_get_prop_prop_not_found() -> None:
    """Test get_prop with non-existent property."""
    with pytest.raises(FramePropError):
        get_prop(clip, "NonExistentProp", int)


def test_get_prop_wrong_type() -> None:
    """Test get_prop with incorrect type specification."""
    with pytest.raises(FramePropError):
        get_prop(clip, "_Matrix", str)


def test_get_prop_fail_to_cast() -> None:
    """Test get_prop with invalid cast function."""
    with pytest.raises(FramePropError):
        get_prop(clip, "_Matrix", int, cast=dict)


def test_get_prop_default() -> None:
    """Test get_prop default value fallback."""
    assert get_prop(clip, "NonExistentProp", int, default=2) == 2


def test_get_prop_func() -> None:
    """Test get_prop with custom function name in error."""
    func_name = "random_function"

    with pytest.raises(FramePropError, match=func_name):
        get_prop(clip, "NonExistentProp", int, func=func_name)


@pytest.mark.parametrize(
    "prop_name, prop_type, expected",
    [
        ("__IntProp", int, 123),
        ("__FloatProp", float, 123),
        ("__BoolProp", int, 1),
    ],
)
def test_get_prop_cast_int_success(prop_name: str, prop_type: type, expected: Any) -> None:
    """Test get_prop casting to int (success cases)."""
    assert get_prop(clip, prop_name, prop_type, cast=int) == expected


@pytest.mark.parametrize(
    "prop_name, prop_type",
    [
        ("__StrProp", str),
        ("__BytesProp", bytes),
        ("__VideoFrameProp", vs.VideoFrame),
    ],
)
def test_get_prop_cast_int_fail(prop_name: str, prop_type: type) -> None:
    """Test get_prop casting to int (expected failures)."""
    with pytest.raises(FramePropError):
        get_prop(clip, prop_name, prop_type, cast=int)


@pytest.mark.parametrize(
    "prop_name, prop_type, expected",
    [
        ("__IntProp", int, 123.0),
        ("__FloatProp", float, 123.456),
        ("__BoolProp", int, 1.0),
    ],
)
def test_get_prop_cast_float_success(prop_name: str, prop_type: type, expected: float) -> None:
    """Test get_prop casting to float (success cases)."""
    assert get_prop(clip, prop_name, prop_type, cast=float) == expected


@pytest.mark.parametrize(
    "prop_name, prop_type",
    [
        ("__StrProp", str),
        ("__BytesProp", bytes),
        ("__VideoFrameProp", vs.VideoFrame),
    ],
)
def test_get_prop_cast_float_fail(prop_name: str, prop_type: type) -> None:
    """Test get_prop casting to float (expected failures)."""
    with pytest.raises(FramePropError):
        get_prop(clip, prop_name, prop_type, cast=float)


@pytest.mark.parametrize(
    "prop_name, prop_type, expected",
    [
        ("__StrProp", str, True),
        ("__IntProp", int, True),
        ("__FloatProp", float, True),
        ("__BoolProp", int, True),
        ("__BytesProp", bytes, True),
        ("__VideoFrameProp", vs.VideoFrame, True),
    ],
)
def test_get_prop_cast_bool(prop_name: str, prop_type: type[Any], expected: Literal[True]) -> None:
    """Test get_prop casting to bool."""
    assert get_prop(clip, prop_name, prop_type, cast=bool) == expected


@pytest.mark.parametrize(
    "prop_name, prop_type, expected",
    [
        ("__StrProp", str, "test string"),
        ("__IntProp", int, "123"),
        ("__FloatProp", float, "123.456"),
        ("__BoolProp", int, "1"),
        ("__BytesProp", bytes, "b'test bytes'"),
        ("__VideoFrameProp", vs.VideoFrame, str(video_frame)),
    ],
)
def test_get_prop_cast_str(prop_name: str, prop_type: type[Any], expected: str) -> None:
    """Test get_prop casting to str."""
    assert get_prop(clip, prop_name, prop_type, cast=str) == expected


@pytest.mark.parametrize(
    "prop_name, prop_type, cast, expected",
    [
        ("__StrProp", str, lambda x: bytes(x, "utf-8"), b"test string"),
        ("__BytesProp", bytes, bytes, b"test bytes"),
        ("__IntProp", int, bytes, bytes(123)),
        ("__BoolProp", int, bytes, bytes(1)),
    ],
)
def test_get_prop_cast_bytes_success(prop_name: str, prop_type: type, cast: Any, expected: bytes) -> None:
    """Test get_prop casting to bytes (success cases)."""
    assert get_prop(clip, prop_name, prop_type, cast=cast) == expected


@pytest.mark.parametrize(
    "prop_name, prop_type",
    [
        ("__VideoFrameProp", vs.VideoFrame),
        ("__FloatProp", float),
    ],
)
def test_get_prop_cast_bytes_fail(prop_name: str, prop_type: type) -> None:
    """Test get_prop casting to bytes (expected failures)."""
    with pytest.raises(FramePropError):
        get_prop(clip, prop_name, prop_type, cast=bytes)


def test_get_prop_error_messages() -> None:
    """Test get_prop error message formatting."""
    with pytest.raises(FramePropError, match="not present in props"):
        get_prop(clip, "NonExistent", int)

    with pytest.raises(FramePropError, match="did not contain expected type"):
        get_prop(clip, "__StrProp", int)


def test_get_prop_cast_custom() -> None:
    """Test get_prop with custom casting function."""

    def custom_cast(x: Any) -> str:
        if isinstance(x, bytes):
            x = x.decode("utf-8")
        return f"Custom: {x}"

    assert get_prop(clip, "__StrProp", str, cast=custom_cast) == "Custom: test string"
    assert get_prop(clip, "__IntProp", int, cast=custom_cast) == "Custom: 123"
    assert get_prop(clip, "__FloatProp", float, cast=custom_cast) == "Custom: 123.456"
    assert get_prop(clip, "__BoolProp", int, cast=custom_cast) == "Custom: 1"
    assert get_prop(clip, "__BytesProp", bytes, cast=custom_cast) == "Custom: test bytes"
    assert get_prop(clip, "__VideoFrameProp", vs.VideoFrame, cast=custom_cast) == f"Custom: {video_frame}"


def test_merge_clip_props_basic() -> None:
    """Test merge_clip_props."""
    merged = merge_clip_props(clip, clip2)

    assert get_prop(merged, "_Matrix", int) == 5
    assert get_prop(merged, "__FloatProp", float) == 123.456
    assert get_prop(merged, "_RandomProp", int) == 1
    assert get_prop(merged, "__AnotherRandomProp", str) == "gsdgsdgs"


def test_merge_clip_props_main_idx() -> None:
    """Test merge_clip_props with main_idx parameter."""
    merged = merge_clip_props(clip, clip2, main_idx=1)

    assert get_prop(merged, "_Matrix", int) == 1
    assert get_prop(merged, "__FloatProp", float) == 123.456
    assert get_prop(merged, "_RandomProp", int) == 1
    assert get_prop(merged, "__AnotherRandomProp", str) == "gsdgsdgs"
