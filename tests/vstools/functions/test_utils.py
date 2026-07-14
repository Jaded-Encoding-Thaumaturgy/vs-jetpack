import pytest

from vstools import (
    UnsupportedColorFamilyError,
    core,
    depth,
    get_b,
    get_g,
    get_r,
    get_u,
    get_v,
    get_y,
    plane,
    stack_planes,
    vs,
)


def test_depth() -> None:
    src_8 = core.std.BlankClip(format=vs.YUV420P8)
    src_10 = depth(src_8, 10)
    assert src_10.format
    assert src_10.format.name == "YUV420P10"

    src2_10 = core.std.BlankClip(format=vs.RGB30)
    src2_8 = depth(src2_10, 8)
    assert src2_8.format
    assert src2_8.format.name == "RGB24"


def test_get_y() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8)
    result = get_y(clip)
    assert result.format
    assert result.format.name == "Gray8"


def test_get_y_invalid() -> None:
    clip = core.std.BlankClip(format=vs.RGB24)
    with pytest.raises(UnsupportedColorFamilyError):
        get_y(clip)


def test_get_u() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8)
    result = get_u(clip)
    assert result.format
    assert result.format.name == "Gray8"


def test_get_u_invalid() -> None:
    clip = core.std.BlankClip(format=vs.RGB24)
    with pytest.raises(UnsupportedColorFamilyError):
        get_u(clip)


def test_get_v() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8)
    result = get_v(clip)
    assert result.format
    assert result.format.name == "Gray8"


def test_get_v_invalid() -> None:
    clip = core.std.BlankClip(format=vs.RGB24)
    with pytest.raises(UnsupportedColorFamilyError):
        get_v(clip)


def test_get_r() -> None:
    clip = core.std.BlankClip(format=vs.RGB24)
    result = get_r(clip)
    assert result.format
    assert result.format.name == "Gray8"


def test_get_r_invalid() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8)
    with pytest.raises(UnsupportedColorFamilyError):
        get_r(clip)


def test_get_g() -> None:
    clip = core.std.BlankClip(format=vs.RGB24)
    result = get_g(clip)
    assert result.format
    assert result.format.name == "Gray8"


def test_get_g_invalid() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8)
    with pytest.raises(UnsupportedColorFamilyError):
        get_g(clip)


def test_get_b() -> None:
    clip = core.std.BlankClip(format=vs.RGB24)
    result = get_b(clip)
    assert result.format
    assert result.format.name == "Gray8"


def test_get_b_invalid() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8)
    with pytest.raises(UnsupportedColorFamilyError):
        get_b(clip)


def test_plane() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8)
    result = plane(clip, 0)
    assert result.format
    assert result.format.name == "Gray8"


# Remove 32-bit integer formats since resize doesn't support the conversion to it.
def get_formats() -> list[vs.PresetVideoFormat]:
    return [fmt for fmt in vs.PresetVideoFormat if fmt != vs.PresetVideoFormat.NONE and not fmt.name.endswith("32")]


@pytest.mark.parametrize("fmt", get_formats(), ids=lambda fmt: f"{fmt.name}")
def test_stack_plane(fmt: vs.PresetVideoFormat) -> None:
    stack_planes(core.std.BlankClip(format=fmt))
