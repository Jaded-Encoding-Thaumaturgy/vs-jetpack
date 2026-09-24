from fractions import Fraction

import pytest

from vsdenoise import AnalyzeArgs, MaskMode, MVTools, SearchMode, SharpMode, SuperArgs
from vsdenoise.mvtools.mvtools import _super_clip_cache
from vstools import UnsupportedColorFamilyError, core, vs


def test_mvtools_init() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8, width=160, height=120)
    mv = MVTools(clip)
    assert mv.clip is clip
    assert mv.chroma is None

    mv_search = MVTools(clip, search_clip=lambda clip: clip.std.Invert())
    assert mv_search.search_clip is not clip

    rgb_clip = core.std.BlankClip(format=vs.RGB24, width=160, height=120)
    with pytest.raises(UnsupportedColorFamilyError):
        MVTools(rgb_clip)

    mv_custom = MVTools(
        clip,
        chroma=True,
        super_args=SuperArgs(sharp=SharpMode.BICUBIC),
        analyze_args=AnalyzeArgs(search=SearchMode.HEXAGON),
    )
    assert mv_custom.chroma is True
    assert mv_custom.super_args["sharp"] == 1


def test_mvtools_super() -> None:
    clip = core.std.BlankClip(format=vs.GRAY8, width=160, height=120)
    mv = MVTools(clip)

    mv.vectors.blksize = (16, 16)
    mv.vectors.overlap = (8, 8)

    # Test default super
    s = mv.super()
    f = s.get_frame(0)
    assert f.props["MVUtensilsSuperPel"] == 2
    assert f.props["MVUtensilsSuperHPad"] == 16
    assert f.props["MVUtensilsSuperVPad"] == 16

    # Test super with pelclip callable
    s_pel = mv.super(
        pelclip=lambda clip: clip.std.BlankClip(width=clip.width * 2, height=clip.height * 2),
        onelevel=True,
    )
    f_pel = s_pel.get_frame(0)
    assert f_pel.props["MVUtensilsSuperPel"] == 2
    assert f_pel.props["MVUtensilsSuperLevels"] == 1


def test_mvtools_analyze() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8, width=160, height=120, length=5)
    mv = MVTools(clip)

    # Default analyze (tr=1)
    mv.analyze(tr=1)
    assert len(mv.vectors) == 2

    # Verify vector properties
    f = mv.vectors[1].get_frame(0)
    assert f.props["MVUtensilsAnalysisDeltaFrame"] == 1
    assert f.props["MVUtensilsAnalysisBlkSizeX"] == 16
    assert f.props["MVUtensilsAnalysisBlkSizeY"] == 16

    # Analyze with custom delta list
    mv.analyze(delta=[-1, 1])
    assert 1 in mv.vectors
    assert -1 in mv.vectors

    f_d2 = mv.vectors[-1].get_frame(0)
    assert f_d2.props["MVUtensilsAnalysisDeltaFrame"] == -1


def test_mvtools_recalculate() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8, width=160, height=120, length=5)
    mv = MVTools(clip)
    mv.analyze(tr=1)

    # Recalculate vectors with a custom block size to verify wrapper parameter passing
    mv.recalculate(blksize=8)
    assert len(mv.vectors) == 2

    f = mv.vectors[1].get_frame(0)
    assert f.props["MVUtensilsAnalysisBlkSizeX"] == 8
    assert f.props["MVUtensilsAnalysisBlkSizeY"] == 8


def test_mvtools_compensate() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8, width=160, height=120, length=5)
    mv = MVTools(clip)
    mv.analyze(tr=1)

    # Default compensate with interleave=True
    comp_clip, (cycle, offset) = mv.compensate(interleave=True)
    assert cycle == 3
    assert offset == 1
    assert comp_clip.num_frames == 15

    # Compensate with interleave=False
    comp_fwrd, comp_back = mv.compensate(interleave=False)
    assert len(comp_fwrd) == 1
    assert len(comp_back) == 1
    assert comp_fwrd[0].num_frames == 5
    assert comp_back[0].num_frames == 5

    # Compensate with temporal_func
    comp_temp = mv.compensate(temporal_func=lambda clip: clip.std.Invert())
    assert comp_temp.num_frames == 5


def test_mvtools_flow() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8, width=160, height=120, length=5)
    mv = MVTools(clip)
    mv.analyze(tr=1)

    # Default flow with interleave=True
    flow_clip, (cycle, offset) = mv.flow(interleave=True)
    assert cycle == 3
    assert offset == 1
    assert flow_clip.num_frames == 15

    # Flow with interleave=False
    flow_fwrd, flow_back = mv.flow(interleave=False)
    assert len(flow_fwrd) == 1
    assert len(flow_back) == 1
    assert flow_fwrd[0].num_frames == 5
    assert flow_back[0].num_frames == 5

    # Flow with temporal_func
    flow_temp = mv.flow(temporal_func=lambda clip: clip.std.Invert())
    assert flow_temp.num_frames == 5


def test_mvtools_degrain() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8, width=160, height=120, length=5)
    mv = MVTools(clip)
    mv.analyze(tr=2)

    degrained = mv.degrain(tr=2, thsad=400, thsad2=200, limit=5.0, weights=[1, 1, 1, 1, 1])
    assert degrained.num_frames == 5
    assert degrained.format == clip.format


def test_mvtools_flow_interpolate() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8, width=160, height=120, length=5)
    mv = MVTools(clip)
    mv.analyze(tr=1)

    # Flow interpolate with interleave=True
    interp_clip = mv.flow_interpolate(interleave=True)
    assert interp_clip.num_frames == 10

    # Flow interpolate with interleave=False
    interp_clip_no_interleave = mv.flow_interpolate(interleave=False)
    assert interp_clip_no_interleave.num_frames == 5


def test_mvtools_flow_fps() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8, width=160, height=120, length=5, fpsnum=24, fpsden=1)
    mv = MVTools(clip)
    mv.analyze(tr=1)

    # Flow FPS with target FPS
    fps_clip = mv.flow_fps(fps=Fraction(30, 1))
    assert fps_clip.fps == Fraction(30, 1)
    assert fps_clip.num_frames == 6


def test_mvtools_flow_blur() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8, width=160, height=120, length=5)
    mv = MVTools(clip)
    mv.analyze(tr=1)

    blurred = mv.flow_blur()
    assert blurred.num_frames == 5
    assert blurred.format == clip.format


def test_mvtools_mask() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8, width=160, height=120, length=5)
    mv = MVTools(clip)
    mv.analyze(tr=1)

    m1 = mv.mask(kind=MaskMode.VECTOR_LENGTH)
    assert m1.format.color_family == vs.GRAY
    m2 = mv.mask(kind=MaskMode.SAD)
    assert m2.format.color_family == vs.GRAY
    m3 = mv.mask(kind=MaskMode.OCCLUSION)
    assert m3.format.color_family == vs.GRAY


def test_mvtools_sc_detection() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8, width=160, height=120, length=5)
    mv = MVTools(clip)
    mv.analyze(tr=1)

    sc_clip = mv.sc_detection()
    assert sc_clip.num_frames == 5
    assert sc_clip.format == clip.format


def test_mvtools_super_cache_reuse() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8, width=160, height=120, length=5)
    mv = MVTools(clip)

    mv.analyze(blksize=16, overlap_div=2)
    cached_super = _super_clip_cache._cache[mv.search_clip]
    assert len(set(cached_super.values())) == 1

    mv.recalculate(blksize=8, overlap_div=2)
    assert len(set(cached_super.values())) == 1

    degrained = mv.degrain()
    assert degrained.num_frames == 5
    assert len(set(cached_super.values())) == 1
