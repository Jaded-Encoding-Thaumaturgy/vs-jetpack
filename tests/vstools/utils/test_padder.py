import pytest
from jetpytools import CustomValueError

from vstools import Align, Padder, core, get_lowest_values, get_neutral_values, get_peak_values, vs


def test_padder_sequence_and_len() -> None:
    pad = Padder(1, 2, 3, 4)
    assert len(pad) == 4
    assert pad[0] == 1
    assert pad[1] == 2
    assert pad[2] == 3
    assert pad[3] == 4
    assert list(pad) == [1, 2, 3, 4]


def test_padder_missing_dimensions() -> None:
    pad = Padder(1, 2, 3, 4)
    with pytest.raises(CustomValueError):
        pad.padded_width
    with pytest.raises(CustomValueError):
        pad.padded_height


def test_padder_crop_scaling() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8, width=1920, height=1080)
    pad = Padder.from_mod(clip, mod=16, min=4, align=Align.MIDDLE_CENTER)
    padded = pad.mirror(clip)

    scaled_padded = core.std.BlankClip(format=vs.YUV420P8, width=padded.width * 2, height=padded.height * 2)
    cropped = pad.crop(scaled_padded)
    assert cropped.width == clip.width * 2
    assert cropped.height == clip.height * 2


def test_padder_crop_float_scale() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8, width=1920, height=1080)
    pad = Padder(2, 2, 2, 2, clip.width, clip.height)
    padded = pad.mirror(clip)

    # 2x scale
    scaled_padded = core.std.BlankClip(format=vs.YUV420P8, width=padded.width * 2, height=padded.height * 2)
    cropped = pad.crop(scaled_padded, crop_scale=2.0)
    assert cropped.width == clip.width * 2
    assert cropped.height == clip.height * 2


def test_padder_explicit() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8, width=1920, height=1080)
    pad = Padder(left=2, right=4, top=6, bottom=8)

    padded = pad.mirror(clip)
    assert padded.width == 1920 + 2 + 4
    assert padded.height == 1080 + 6 + 8


def test_subsampling_error() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8, width=1920, height=1080)
    pad = Padder(left=1, right=0, top=0, bottom=0)
    with pytest.raises(CustomValueError):
        pad.mirror(clip)


def test_padder_color_default() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8, width=10, height=10).std.SetFrameProp("_Range", vs.RANGE_LIMITED)
    lowests = get_lowest_values(clip)
    neutrals = get_neutral_values(clip)
    pad = Padder(2, 2, 2, 2, clip.width, clip.height)

    padded = pad.color(clip).get_frame(0)
    assert padded[0][0, 0] == lowests[0]
    assert padded[1][0, 0] == neutrals[1]
    assert padded[2][0, 0] == neutrals[2]


def test_padder_color_constant() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8, width=10, height=10).std.SetFrameProp("_Range", vs.RANGE_LIMITED)
    pad = Padder(2, 2, 2, 2, clip.width, clip.height)

    color = 4
    padded = pad.color(clip, color=color).get_frame(0)
    for i in range(clip.format.num_planes):
        assert padded[i][0, 0] == color


def test_padder_color_sequence() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8, width=10, height=10).std.SetFrameProp("_Range", vs.RANGE_LIMITED)
    pad = Padder(2, 2, 2, 2, clip.width, clip.height)

    color = [1, 2, 3]
    padded = pad.color(clip, color=color).get_frame(0)
    for i, expected in zip(range(clip.format.num_planes), color):
        assert padded[i][0, 0] == expected


def test_padder_color_boolean() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8, width=10, height=10).std.SetFrameProp("_Range", vs.RANGE_LIMITED)
    peaks = get_peak_values(clip)
    lowests = get_lowest_values(clip)
    pad = Padder(2, 2, 2, 2, clip.width, clip.height)

    padded_true = pad.color(clip, color=True).get_frame(0)
    for i in range(clip.format.num_planes):
        assert padded_true[i][0, 0] == peaks[i]

    padded_false = pad.color(clip, color=False).get_frame(0)
    for i in range(clip.format.num_planes):
        assert padded_false[i][0, 0] == lowests[i]


def test_padder_color_neutral() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8, width=10, height=10).std.SetFrameProp("_Range", vs.RANGE_LIMITED)
    neutrals = get_neutral_values(clip)
    pad = Padder(2, 2, 2, 2, clip.width, clip.height)

    padded = pad.color(clip, color=None).get_frame(0)
    for i in range(clip.format.num_planes):
        assert padded[i][0, 0] == neutrals[i]


def test_padder_from_mod() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8, width=1920, height=1080)

    pad = Padder.from_mod(clip, mod=16, min=4, align=Align.MIDDLE_CENTER)
    assert pad.left >= 4
    assert pad.right >= 4
    assert pad.top >= 4
    assert pad.bottom >= 4
    assert (clip.width + pad.left + pad.right) % 16 == 0
    assert (clip.height + pad.top + pad.bottom) % 16 == 0


def test_padder_alignments() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8, width=1920, height=1080)

    # Test different align flags
    p1 = Padder.from_mod(clip, mod=16, min=4, align=Align.TOP_LEFT)
    assert p1.top == 0
    assert p1.left == 0

    p2 = Padder.from_mod(clip, mod=16, min=4, align=Align.BOTTOM_RIGHT)
    assert p2.bottom == 0
    assert p2.right == 0


def test_mod_padding_crop_scale() -> None:
    padding, crop_padding = Padder.mod_padding_crop((1920, 1080), mod=16, min=4, crop_scale=2.0)
    assert crop_padding[0] == padding[0] * 2
