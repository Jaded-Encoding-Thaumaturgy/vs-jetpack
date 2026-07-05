import pytest

from vstools import Matrix, Primaries, Transfer, finalize_clip, get_prop, initialize_clip, vs

pytestmark = pytest.mark.vpy("initial-core", "reloaded-core")


def test_finalize_clip() -> None:
    clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=1920, height=1080)
    clip = finalize_clip(clip)
    assert clip.format
    assert clip.format.bits_per_sample == 10

    clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=1920, height=1080)
    clip = finalize_clip(clip, bits=16)
    assert clip.format
    assert clip.format.bits_per_sample == 16

    clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=1920, height=1080)
    clip = finalize_clip(clip, bits=None)
    assert clip.format
    assert clip.format.bits_per_sample == 8


@pytest.mark.filterwarnings("ignore:The _ColorRange frame property has been deprecated, use _Range instead")
def test_initialize_clip() -> None:
    clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=1920, height=1080)
    clip = initialize_clip(clip)
    assert clip.format
    assert clip.format.bits_per_sample == 32
    assert get_prop(clip, "_Matrix", int) == 1
    assert get_prop(clip, "_Primaries", int) == 1
    assert get_prop(clip, "_Transfer", int) == 1

    clip = vs.core.std.BlankClip(format=vs.YUV420P8, width=1920, height=1080)
    clip = initialize_clip(clip, matrix=Matrix.ST170_M, transfer=Transfer.BT470_BG, primaries=Primaries.BT470_BG)
    assert clip.format
    assert clip.format.bits_per_sample == 32
    assert get_prop(clip, "_Matrix", int) == 6
    assert get_prop(clip, "_Primaries", int) == 5
    assert get_prop(clip, "_Transfer", int) == 5
