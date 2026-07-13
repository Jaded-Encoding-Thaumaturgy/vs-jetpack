from fractions import Fraction

import pytest

from vsdenoise import MaskMode, MVDirection, MVTools
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
        pad=(8, 8),
        pel=2,
        chroma=True,
        super_args={"sharp": 1},
        analyze_args={"search": 2},
    )
    assert mv_custom.pad == [8, 8]
    assert mv_custom.pel == 2
    assert mv_custom.chroma is True
    assert mv_custom.super_args["sharp"] == 1


def test_mvtools_super() -> None:
    clip = core.std.BlankClip(format=vs.GRAY8, width=160, height=120)
    mv = MVTools(clip)

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
    assert len(mv.vectors[MVDirection.BACKWARD]) == 1
    assert len(mv.vectors[MVDirection.FORWARD]) == 1

    # Verify vector properties
    f = mv.vectors[MVDirection.BACKWARD][1].get_frame(0)
    assert f.props["MVUtensilsAnalysisDeltaFrame"] == 1
    assert f.props["MVUtensilsAnalysisBlkSizeX"] == 16
    assert f.props["MVUtensilsAnalysisBlkSizeY"] == 16

    # Analyze with custom delta list
    mv.analyze(delta=[1, 2])
    assert 2 in mv.vectors[MVDirection.BACKWARD]
    assert 2 in mv.vectors[MVDirection.FORWARD]

    f_d2 = mv.vectors[MVDirection.FORWARD][2].get_frame(0)
    assert f_d2.props["MVUtensilsAnalysisDeltaFrame"] == -2


def test_mvtools_recalculate() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8, width=160, height=120, length=5)
    mv = MVTools(clip)
    mv.analyze(tr=1)

    # Recalculate vectors with a custom block size to verify wrapper parameter passing
    mv.recalculate(blksize=8)
    assert len(mv.vectors[MVDirection.BACKWARD]) == 1

    f = mv.vectors[MVDirection.BACKWARD][1].get_frame(0)
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
    comp_back, comp_fwrd = mv.compensate(interleave=False)
    assert len(comp_back) == 1
    assert len(comp_fwrd) == 1
    assert comp_back[0].num_frames == 5
    assert comp_fwrd[0].num_frames == 5

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
    flow_back, flow_fwrd = mv.flow(interleave=False)
    assert len(flow_back) == 1
    assert len(flow_fwrd) == 1
    assert flow_back[0].num_frames == 5
    assert flow_fwrd[0].num_frames == 5

    # Flow with temporal_func
    flow_temp = mv.flow(temporal_func=lambda clip: clip.std.Invert())
    assert flow_temp.num_frames == 5


def test_mvtools_degrain() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8, width=160, height=120, length=5)
    mv = MVTools(clip)
    mv.analyze(tr=1)

    degrained = mv.degrain(tr=1, limit=5.0, weights=[1, 1, 1])
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
