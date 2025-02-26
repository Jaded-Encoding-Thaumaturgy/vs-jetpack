from typing import Any
from unittest import TestCase

from vstools import FramePropError, get_prop, vs, core, merge_clip_props

clip = core.std.BlankClip(
    format=vs.YUV420P8, width=1920, height=1080,
)

clip = clip.std.SetFrameProps(
    _Matrix=1, _Transfer=1, _Primaries=1,
    __StrProp="test string", __IntProp=123, __FloatProp=123.456,
    __BoolProp=True, _BytesProp=b"test bytes",
    __VideoFrameProp=clip.get_frame(0)
)

clip2 = core.std.BlankClip(
    format=vs.YUV420P8, width=1920, height=1080,
)

clip2 = clip2.std.SetFrameProps(
    _Matrix=5, _RandomProp=1, __AnotherRandomProp="gsdgsdgs"
)

class TestGetProp(TestCase):
    """Test cases for the get_prop function."""

    def test_get_prop_video_node_input(self) -> None:
        """Test get_prop with VideoNode input."""

        self.assertEqual(get_prop(clip, "_Matrix", int), 1)
        self.assertEqual(get_prop(clip, "_Transfer", int), 1)
        self.assertEqual(get_prop(clip, "_Primaries", int), 1)

    def test_get_prop_video_frame_input(self) -> None:
        """Test get_prop with VideoFrame input."""

        self.assertEqual(get_prop(clip.get_frame(0), "_Matrix", int), 1)
        self.assertEqual(get_prop(clip.get_frame(0), "_Transfer", int), 1)
        self.assertEqual(get_prop(clip.get_frame(0), "_Primaries", int), 1)

    def test_get_prop_frame_props_input(self) -> None:
        """Test get_prop with FrameProps input."""

        props = clip.get_frame(0).props

        self.assertEqual(get_prop(props, "_Matrix", int), 1)
        self.assertEqual(get_prop(props, "_Transfer", int), 1)
        self.assertEqual(get_prop(props, "_Primaries", int), 1)

    def test_get_prop_prop_not_found(self) -> None:
        """Test get_prop with non-existent property."""

        self.assertRaises(FramePropError, get_prop, clip, "NonExistentProp", int)

    def test_get_prop_wrong_type(self) -> None:
        """Test get_prop with incorrect type specification."""

        self.assertRaises(FramePropError, get_prop, clip, "_Matrix", str)

    def test_get_prop_fail_to_cast(self) -> None:
        """Test get_prop with invalid cast function."""

        self.assertRaises(FramePropError, get_prop, clip, "_Matrix", int, cast=dict)

    def test_get_prop_default(self) -> None:
        """Test get_prop default value fallback."""

        self.assertEqual(get_prop(clip, "_Matrix", int, default=2), 2)

    def test_get_prop_func(self) -> None:
        """Test get_prop with custom function name in error."""

        func_name = "random_function"

        with self.assertRaisesRegex(FramePropError, func_name):
            get_prop(clip, "NonExistentProp", int, func=func_name)

    def test_get_prop_cast_int(self) -> None:
        """Test get_prop casting to int."""

        self.assertEqual(get_prop(clip, "__IntProp", int, cast=int), 123)
        self.assertEqual(get_prop(clip, "__FloatProp", float, cast=int), 123)
        self.assertEqual(get_prop(clip, "__BoolProp", int, cast=int), 1)

        self.assertRaises(FramePropError, get_prop, clip, "__StrProp", str, cast=int)
        self.assertRaises(FramePropError, get_prop, clip, "__BytesProp", bytes, cast=int)
        self.assertRaises(FramePropError, get_prop, clip, "__VideoFrameProp", vs.VideoFrame, cast=int)

    def test_get_prop_cast_float(self) -> None:
        """Test get_prop casting to float."""

        self.assertEqual(get_prop(clip, "__IntProp", int, cast=float), 123.0)
        self.assertEqual(get_prop(clip, "__FloatProp", float, cast=float), 123.456)
        self.assertEqual(get_prop(clip, "__BoolProp", int, cast=float), 1.0)

        self.assertRaises(FramePropError, get_prop, clip, "__StrProp", str, cast=float)
        self.assertRaises(FramePropError, get_prop, clip, "__BytesProp", bytes, cast=float)
        self.assertRaises(FramePropError, get_prop, clip, "__VideoFrameProp", vs.VideoFrame, cast=float)

    def test_get_prop_cast_bool(self) -> None:
        """Test get_prop casting to bool."""

        self.assertEqual(get_prop(clip, "__StrProp", str, cast=bool), True)
        self.assertEqual(get_prop(clip, "__IntProp", int, cast=bool), True)
        self.assertEqual(get_prop(clip, "__FloatProp", float, cast=bool), True)
        self.assertEqual(get_prop(clip, "__BoolProp", int, cast=bool), True)
        self.assertEqual(get_prop(clip, "__BytesProp", bytes, cast=bool), True)
        self.assertEqual(get_prop(clip, "__VideoFrameProp", vs.VideoFrame, cast=bool), True)

    def test_get_prop_cast_str(self) -> None:
        """Test get_prop casting to str."""

        self.assertEqual(get_prop(clip, "__StrProp", str, cast=str), "test string")
        self.assertEqual(get_prop(clip, "__IntProp", int, cast=str), "123")
        self.assertEqual(get_prop(clip, "__FloatProp", float, cast=str), "123.456")
        self.assertEqual(get_prop(clip, "__BoolProp", int, cast=str), "1")
        self.assertEqual(get_prop(clip, "__BytesProp", bytes, cast=str), "test bytes")
        self.assertEqual(get_prop(clip, "__VideoFrameProp", vs.VideoFrame, cast=str), str(clip.get_frame(0)))

    def test_get_prop_cast_bytes(self) -> None:
        """Test get_prop casting to bytes."""

        self.assertEqual(get_prop(clip, "__StrProp", str, cast=bytes), b"test string")
        self.assertEqual(get_prop(clip, "__BytesProp", bytes, cast=bytes), b"test bytes")
        self.assertEqual(get_prop(clip, "__VideoFrameProp", vs.VideoFrame, cast=bytes), bytes(clip.get_frame(0)))
        self.assertRaises(FramePropError, get_prop, clip, "__IntProp", int, cast=bytes)
        self.assertRaises(FramePropError, get_prop, clip, "__FloatProp", float, cast=bytes)
        self.assertRaises(FramePropError, get_prop, clip, "__BoolProp", int, cast=bytes)

    def test_get_prop_error_messages(self) -> None:
        """Test get_prop error message formatting."""

        with self.assertRaisesRegex(FramePropError, "not present in props"):
            get_prop(clip, "NonExistent", int)

        with self.assertRaisesRegex(FramePropError, "did not contain expected type"):
            get_prop(clip, "__StrProp", int)

    def test_get_prop_cast_custom(self) -> None:
        """Test get_prop with custom casting function."""

        def custom_cast(x: Any) -> str:
            return f"Custom: {x}"

        self.assertEqual(get_prop(clip, "__StrProp", str, cast=custom_cast), "Custom: test string")
        self.assertEqual(get_prop(clip, "__IntProp", int, cast=custom_cast), "Custom: 123")
        self.assertEqual(get_prop(clip, "__FloatProp", float, cast=custom_cast), "Custom: 123.456")
        self.assertEqual(get_prop(clip, "__BoolProp", int, cast=custom_cast), "Custom: 1")
        self.assertEqual(get_prop(clip, "__BytesProp", bytes, cast=custom_cast), "Custom: test bytes")
        self.assertEqual(get_prop(clip, "__VideoFrameProp", vs.VideoFrame, cast=custom_cast), f"Custom: {clip.get_frame(0)}")


class TestMergeClipProps(TestCase):
    """Test cases for the merge_clip_props function."""

    def test_merge_clip_props_basic(self) -> None:
        """Test merge_clip_props"""

        merged = merge_clip_props(clip, clip2)

        self.assertEqual(get_prop(merged, "_Matrix", int), 1)
        self.assertEqual(get_prop(merged, "__FloatProp", float), 123.456)
        self.assertEqual(get_prop(merged, "_RandomProp", int), 1)
        self.assertEqual(get_prop(merged, "__AnotherRandomProp", str), "gsdgsdgs")

    def test_merge_clip_props_main_idx(self) -> None:
        """Test merge_clip_props with main_idx parameter."""

        merged = merge_clip_props(clip, clip2, main_idx=1)

        self.assertEqual(get_prop(merged, "_Matrix", int), 5)
        self.assertEqual(get_prop(merged, "__FloatProp", float), 123.456)
        self.assertEqual(get_prop(merged, "_RandomProp", int), 1)
        self.assertEqual(get_prop(merged, "__AnotherRandomProp", str), "gsdgsdgs")
