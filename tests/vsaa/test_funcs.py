from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from jetpytools import CustomTypeError, CustomValueError

from vsaa.deinterlacers import EEDI3, NNEDI3, AntiAliaser
from vsaa.funcs import based_aa, pre_aa
from vsexprtools import norm_expr
from vskernels import Box, Catrom, NoScale, Point
from vstools import UnsupportedColorFamilyError, core, get_y, vs


@pytest.fixture
def edge_clip() -> vs.VideoNode:
    blank = core.std.BlankClip(format=vs.YUV420P8, width=160, height=120, length=5, color=[16, 128, 128])
    clip = norm_expr(blank, "X Y 2 * + 100 > 235 16 ?")
    return clip.std.SetFrameProp("_FieldBased", intval=0)


@pytest.fixture
def edge_clip_gray() -> vs.VideoNode:
    blank = core.std.BlankClip(format=vs.GRAY8, width=160, height=120, length=5, color=16)
    clip = norm_expr(blank, "X Y 2 * + 100 > 235 16 ?")
    return clip.std.SetFrameProp("_FieldBased", intval=0)


def test_based_aa_defaults_yuv(edge_clip: vs.VideoNode) -> None:
    clip = edge_clip
    res = based_aa(clip, supersampler=Point)

    assert res.format.id == clip.format.id
    assert res.width == clip.width
    assert res.height == clip.height
    assert res.num_frames == clip.num_frames

    frame_src = clip.get_frame(0)
    frame_res = res.get_frame(0)
    # YUV output preserves original chroma planes
    assert frame_res[1] == frame_src[1]
    assert frame_res[2] == frame_src[2]
    # Luma plane is modified by AA processing
    assert frame_res[0] != frame_src[0]


def test_based_aa_gray(edge_clip_gray: vs.VideoNode) -> None:
    clip = edge_clip_gray
    res = based_aa(clip, supersampler=Point)

    assert res.format.id == vs.GRAY8
    assert res.format.num_planes == 1
    assert res.width == clip.width
    assert res.height == clip.height

    frame_src = clip.get_frame(0)
    frame_res = res.get_frame(0)
    assert frame_res[0] != frame_src[0]


def test_based_aa_invalid_color_family() -> None:
    clip = core.std.BlankClip(format=vs.RGB24, width=160, height=120, length=5)
    with pytest.raises(UnsupportedColorFamilyError):
        based_aa(clip, supersampler=Point)


def test_based_aa_invalid_rfactor() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8, width=160, height=120, length=5)
    with pytest.raises(CustomValueError):
        based_aa(clip, rfactor=0.0, supersampler=Point)
    with pytest.raises(CustomValueError):
        based_aa(clip, rfactor=-1.0, supersampler=Point)


def test_based_aa_show_mask(edge_clip: vs.VideoNode) -> None:
    clip = edge_clip
    mask_clip = based_aa(clip, show_mask=True, supersampler=Point)

    assert mask_clip.format.id == vs.GRAY8

    frame_mask = mask_clip.get_frame(0)[0]

    # Flat area far from the diagonal, should have mask value 0
    assert frame_mask[5, 5] == 0

    # Near diagonal edge, mask is active (>0)
    assert frame_mask[25, 50] > 0 or frame_mask[25, 51] > 0 or frame_mask[26, 50] > 0


def test_based_aa_custom_mask_node(edge_clip: vs.VideoNode) -> None:
    clip = edge_clip

    # All black mask means MaskedMerge preserves 100% original luma
    black_mask = clip.std.BlankClip(format=vs.GRAY8, color=0)
    res_black = based_aa(clip, mask=black_mask, supersampler=Point)

    frame_src = clip.get_frame(0)
    frame_black = res_black.get_frame(0)
    assert frame_black[0] == frame_src[0]

    # All white mask means MaskedMerge uses 100% AA clip, matching mask=False
    white_mask = clip.std.BlankClip(format=vs.GRAY8, color=255)
    res_white = based_aa(clip, mask=white_mask, supersampler=Point)
    res_nomask = based_aa(clip, mask=False, supersampler=Point)

    assert res_white.get_frame(0)[0] == res_nomask.get_frame(0)[0]


def test_based_aa_mask_false(edge_clip: vs.VideoNode) -> None:
    clip = edge_clip

    # With prefilter inverting luma, mask=False inverts full frame while mask=Prewitt restricts inversion to edges
    res_nomask = based_aa(clip, mask=False, prefilter=core.std.Invert, supersampler=Point)
    res_masked = based_aa(clip, prefilter=core.std.Invert, supersampler=Point)

    frame_src = clip.get_frame(0)
    frame_nomask = res_nomask.get_frame(0)
    frame_masked = res_masked.get_frame(0)

    # In flat area far from edge, masked AA keeps original pixel value (16)
    assert frame_masked[0][0, 0] == frame_src[0][0, 0] == 16
    # While mask=False inverts full frame (239)
    assert frame_nomask[0][0, 0] == 239

    assert frame_nomask[0] != frame_masked[0]


def test_based_aa_supersampler_false(edge_clip: vs.VideoNode) -> None:
    clip = edge_clip
    # Passing supersampler=False forces rfactor=1.0 and sets supersampler/downscaler to NoScale
    res_false = based_aa(clip, rfactor=4.0, pscale=0.8, supersampler=False)
    res_noscale = based_aa(clip, rfactor=1.0, pscale=1.0, supersampler=NoScale)

    assert res_false.get_frame(0)[0] == res_noscale.get_frame(0)[0]


def test_based_aa_downscaler_none_rfactors(edge_clip: vs.VideoNode) -> None:
    clip = edge_clip

    # Integer factor (2.0) -> downscaler is automatically set to Box
    res_int = based_aa(clip, rfactor=2.0, supersampler=Point, downscaler=None)
    expected_box = based_aa(clip, rfactor=2.0, supersampler=Point, downscaler=Box)
    assert res_int.get_frame(0)[0] == expected_box.get_frame(0)[0]

    # Non-integer factor (1.5) -> downscaler is automatically set to Catrom
    res_float = based_aa(clip, rfactor=1.5, supersampler=Point, downscaler=None)
    expected_catrom = based_aa(clip, rfactor=1.5, supersampler=Point, downscaler=Catrom)
    assert res_float.get_frame(0)[0] == expected_catrom.get_frame(0)[0]


def test_based_aa_rfactor_less_than_one(edge_clip: vs.VideoNode) -> None:
    clip = edge_clip

    # When rfactor < 1.0, supersampler and downscaler are swapped internally:
    # supersampler=Point, downscaler=Box with rfactor=0.5 -> downscales with Box first, supersamples with Point after
    res_sub = based_aa(clip, rfactor=0.5, supersampler=Point, downscaler=Box)

    # Downscaling with Point first and supersampling with Box after produces different output
    res_opposite = based_aa(clip, rfactor=0.5, supersampler=Box, downscaler=Point)

    assert res_sub.get_frame(0)[0] != res_opposite.get_frame(0)[0]


def test_based_aa_prefilter_paths(edge_clip: vs.VideoNode) -> None:
    clip = edge_clip

    prefilter_calls = []

    def spy_prefilter(clip: vs.VideoNode) -> vs.VideoNode:
        prefilter_calls.append(clip)
        return clip.std.Invert()

    res_func = based_aa(clip, prefilter=spy_prefilter, supersampler=Point)
    assert len(prefilter_calls) == 1
    assert prefilter_calls[0].width == clip.width

    inverted_clip = clip.std.Invert()
    res_node = based_aa(clip, prefilter=inverted_clip, supersampler=Point)

    # Both prefilters invert luma before AA so their outputs match each other
    assert res_func.get_frame(0)[0] == res_node.get_frame(0)[0]

    # And differ from default
    res_default = based_aa(clip, prefilter=False, supersampler=Point)
    assert res_func.get_frame(0)[0] != res_default.get_frame(0)[0]


@dataclass
class MockAntiAliaser(NNEDI3):
    called: bool = False
    called_clip: vs.VideoNode | None = None

    def antialias(
        self,
        clip: vs.VideoNode,
        direction: NNEDI3.AADirection = NNEDI3.AADirection.BOTH,
        **kwargs: Any,
    ) -> vs.VideoNode:
        self.called = True
        self.called_clip = clip
        return clip.std.Invert()


def test_based_aa_custom_antialiaser(edge_clip: vs.VideoNode) -> None:
    clip = edge_clip
    mock_aa = MockAntiAliaser()

    res = based_aa(clip, antialiaser=mock_aa, supersampler=Point, rfactor=2.0)

    # Verify custom antialiaser was called once with the supersampled clip (320x240)
    assert mock_aa.called is True
    assert mock_aa.called_clip is not None
    assert mock_aa.called_clip.width == clip.width * 2
    assert mock_aa.called_clip.height == clip.height * 2
    assert res.format.id == clip.format.id


@dataclass
class MockEEDI3(EEDI3):
    last_kwargs: dict[str, Any] | None = None

    def antialias(
        self,
        clip: vs.VideoNode,
        direction: AntiAliaser.AADirection = AntiAliaser.AADirection.BOTH,
        **kwargs: Any,
    ) -> vs.VideoNode:
        self.last_kwargs = kwargs
        return clip


def test_based_aa_eedi3_mclip_paths(edge_clip: vs.VideoNode) -> None:
    clip = edge_clip
    mock_eedi3 = MockEEDI3()

    # rfactor == 1 with mask: mclip passed to EEDI3 is unresized mask matching original dimensions
    based_aa(clip, antialiaser=mock_eedi3, rfactor=1.0, supersampler=NoScale())
    assert mock_eedi3.last_kwargs is not None
    assert mock_eedi3.last_kwargs["mclip"] is not None
    assert mock_eedi3.last_kwargs["mclip"].width == clip.width
    assert mock_eedi3.last_kwargs["mclip"].height == clip.height

    # rfactor == 2 with mask: mclip passed to EEDI3 is resized to supersampled size (320x240)
    based_aa(clip, antialiaser=mock_eedi3, rfactor=2.0, supersampler=Point, downscaler=Point)
    assert mock_eedi3.last_kwargs is not None
    assert mock_eedi3.last_kwargs["mclip"] is not None
    assert mock_eedi3.last_kwargs["mclip"].width == clip.width * 2
    assert mock_eedi3.last_kwargs["mclip"].height == clip.height * 2

    # use_mclip=False suppresses passing mclip to EEDI3
    based_aa(clip, antialiaser=mock_eedi3, use_mclip=False, supersampler=Point)
    assert mock_eedi3.last_kwargs.get("mclip") is None

    # Explicit sclip in aa_kwargs bypasses auto sclip setting (using inverted sclip)
    eedi3 = EEDI3()
    custom_sclip = get_y(clip).std.Invert().resize.Bilinear(clip.width * 2, clip.height * 2)
    res_custom_sclip = based_aa(clip, antialiaser=eedi3, sclip=custom_sclip, supersampler=Point)
    res_default_sclip = based_aa(clip, antialiaser=eedi3, supersampler=Point)

    # Custom sclip produces different pixels than default auto-injected sclip
    assert res_custom_sclip.get_frame(0)[0] != res_default_sclip.get_frame(0)[0]


def test_based_aa_pscale(edge_clip: vs.VideoNode) -> None:
    clip = edge_clip

    res_p0 = based_aa(clip, pscale=0.0, supersampler=Catrom)
    res_p1 = based_aa(clip, pscale=1.0, supersampler=Catrom)

    assert res_p0.get_frame(0)[0] != res_p1.get_frame(0)[0]


def test_based_aa_postfilter_paths(
    edge_clip: vs.VideoNode,
    edge_clip_gray: vs.VideoNode,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clip = edge_clip

    def pf3(aa: vs.VideoNode, ss: vs.VideoNode, luma: vs.VideoNode) -> vs.VideoNode:
        return luma

    res3 = based_aa(clip, postfilter=pf3, supersampler=Point, mask=False)
    # Since pf3 returns luma, output luma equals original clip luma when mask=False
    assert res3.get_frame(0)[0] == clip.get_frame(0)[0]

    def pf1(aa: vs.VideoNode) -> vs.VideoNode:
        return aa.std.Invert()

    res1 = based_aa(clip, postfilter=pf1, supersampler=Point, mask=False)
    res_false = based_aa(clip, postfilter=False, supersampler=Point, mask=False)
    # Inverted postfilter output matches invert of postfilter=False
    assert res1.get_frame(0)[0] == res_false.std.Invert().get_frame(0)[0]

    # Invalid parameter count (e.g. 2 parameters)
    def pf2(aa: vs.VideoNode, ss: vs.VideoNode) -> vs.VideoNode:
        return aa

    with pytest.raises(CustomTypeError):
        based_aa(clip, postfilter=pf2, supersampler=Point)  # type: ignore[arg-type]

    # Dictionary postfilter
    clip = edge_clip_gray
    pfclip = clip.std.BlankClip()
    sigmas = None

    def spy_postfilter(*args: Any, sigmaS: float) -> vs.VideoNode:  # noqa: N803
        nonlocal sigmas
        sigmas = sigmaS
        return pfclip

    monkeypatch.setattr(based_aa, "postfilter", spy_postfilter)
    res_dict = based_aa(clip, postfilter={"sigmaS": 10.0}, supersampler=Point, mask=False)
    assert res_dict == pfclip
    assert sigmas == 10.0


def test_pre_aa(edge_clip: vs.VideoNode) -> None:
    clip = edge_clip

    res_default = pre_aa(clip)
    res_horiz = pre_aa(clip, transpose_first=True, direction=AntiAliaser.AADirection.HORIZONTAL)
    res_vert = pre_aa(clip, direction=AntiAliaser.AADirection.VERTICAL)

    assert res_default.get_frame(0)[0] != res_horiz.get_frame(0)[0]
    assert res_horiz.get_frame(0)[0] != res_vert.get_frame(0)[0]
