from __future__ import annotations

import pytest

from vsdenoise import DFTTest, deblock_qed
from vsdenoise.nlm import NLMeans
from vstools import core, vs


def test_deblock_qed_interlaced() -> None:
    # Create a simple clip
    clip = core.std.BlankClip(width=320, height=240, format=vs.YUV420P8)

    # If deblock plugin is not loaded, we expect AttributeError from deblock_qed.
    # Otherwise, it should succeed and return the correct height.
    if not hasattr(core, "deblock"):
        pytest.skip("VapourSynth 'deblock' plugin not available")

    res = deblock_qed(clip, interlaced=True)
    assert res.height == 240


def test_dfttest_backend_rebuild() -> None:
    # Test configured backend member identity and equality
    old_configured = DFTTest.Backend.OLD(opt=2)
    assert old_configured == DFTTest.Backend.OLD

    cpu_configured = DFTTest.Backend.CPU(opt=3)
    assert cpu_configured == DFTTest.Backend.CPU

    # Verify .value remains a string
    assert isinstance(cpu_configured.value, str)
    assert cpu_configured.value == "dfttest2_cpu"

    # Verify we can resolve without TypeError
    resolved = cpu_configured.resolve()
    assert resolved.value == "dfttest2_cpu"


def test_nlm_weight_mode() -> None:
    wmode = NLMeans.WeightMode.WELSCH(wref=0.5)
    assert wmode == NLMeans.WeightMode.WELSCH
    assert wmode.wref == 0.5
    assert isinstance(wmode.value, int)


def test_bm3d_tr0_old_backend() -> None:
    from vsdenoise import bm3d

    if not hasattr(core, "bm3d"):
        pytest.skip("VapourSynth 'bm3d' plugin not available")

    clip = core.std.BlankClip(width=320, height=240, format=vs.GRAY16)
    res = bm3d(clip, tr=0, backend=bm3d.Backend.OLD)
    assert res.format.color_family == vs.GRAY


def test_bm3d_subsampled_yuv_old_backend_error() -> None:
    from vsdenoise import bm3d
    from vstools import UnsupportedVideoFormatError

    clip = core.std.BlankClip(width=320, height=240, format=vs.YUV420P16)
    with pytest.raises(UnsupportedVideoFormatError):
        bm3d(clip, tr=0, backend=bm3d.Backend.OLD)


def test_mvtools_explicit_vectors_padding() -> None:
    from vsdenoise import MVTools, MotionVectors, MVDirection
    from vsdenoise.mvtools.mvtools import _super_clip_cache
    from jetpytools import cachedproperty
    from types import MappingProxyType

    if not hasattr(core, "mv"):
        pytest.skip("VapourSynth 'mvtools' plugin not available")

    clip = core.std.BlankClip(width=320, height=240, format=vs.YUV420P8)
    mv = MVTools(clip)

    # Prepare custom vectors that are scaled and specify distinct padding (e.g. 32)
    custom_vectors = MotionVectors()
    cachedproperty.update_cache(custom_vectors, "scaled", True)
    cachedproperty.update_cache(
        custom_vectors, "analysis_data", MappingProxyType({"Analysis_Padding": (32, 32)})
    )

    # Populate dummy analyzed vectors in custom_vectors so degrain can read them
    mv.analyze()
    custom_vectors.set_vector(mv.vectors.get_vector(MVDirection.BACKWARD, 1), MVDirection.BACKWARD, 1)
    custom_vectors.set_vector(mv.vectors.get_vector(MVDirection.FORWARD, 1), MVDirection.FORWARD, 1)

    # Clear super clip cache to ensure we get a clean check
    _super_clip_cache.clear()

    # Call degrain with explicit custom vectors
    mv.degrain(vectors=custom_vectors, tr=1)

    # Check that the super clip was cached with hpad=32 and vpad=32
    assert clip in _super_clip_cache
    cache = _super_clip_cache[clip]
    keys = list(cache.keys())
    assert len(keys) > 0
    # Inspect the keys to verify that hpad=32 and vpad=32 were used
    found_correct_padding = False
    for key in keys:
        args_dict = dict(key.args)
        if args_dict.get("hpad") == 32 and args_dict.get("vpad") == 32:
            found_correct_padding = True
            break

    assert found_correct_padding, f"Expected padding of 32 in cached super clip arguments, got: {keys}"


