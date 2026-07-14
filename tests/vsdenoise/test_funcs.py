from vsdenoise import MVDirection, MVTools, mc_clamp, mc_degrain
from vstools import core, vs


def test_mc_degrain_defaults() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8, width=160, height=120, length=5)
    denoised = mc_degrain(clip)
    assert denoised.format.id == clip.format.id
    assert denoised.width == clip.width
    assert denoised.height == clip.height
    assert denoised.num_frames == clip.num_frames


def test_mc_degrain_export_globals() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8, width=160, height=120, length=5)
    denoised, mv = mc_degrain(clip, export_globals=True)
    assert denoised.format.id == clip.format.id
    assert mv.clip is clip
    assert mv.pel is None


def test_mc_degrain_refine() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8, width=160, height=120, length=5)
    # Using refine=2, so block size is refined from 16 to 8 to 4
    denoised, mv = mc_degrain(clip, refine=2, thsad_recalc=200, export_globals=True)
    assert denoised.format.id == clip.format.id

    f = mv.vectors[MVDirection.BACKWARD][1].get_frame(0)
    assert f.props["MVUtensilsAnalysisBlkSizeX"] == 4
    assert f.props["MVUtensilsAnalysisBlkSizeY"] == 4


def test_mc_degrain_pre_calculated_vectors() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8, width=160, height=120, length=5)
    mv = MVTools(clip)
    mv.analyze(tr=1)

    denoised = mc_degrain(clip, vectors=mv.vectors)
    assert denoised.format.id == clip.format.id
    assert denoised.num_frames == clip.num_frames


def test_mc_degrain_filters() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8, width=160, height=120, length=5)

    denoised = mc_degrain(clip, prefilter=clip.std.Invert(), mfilter=clip.std.Invert())
    assert denoised.format.id == clip.format.id

    denoised_call = mc_degrain(clip, prefilter=lambda clip: clip.std.Invert(), mfilter=lambda clip: clip.std.Invert())
    assert denoised_call.format.id == clip.format.id


def test_mc_degrain_params() -> None:
    clip = core.std.BlankClip(format=vs.YUV420P8, width=160, height=120, length=5)
    denoised = mc_degrain(
        clip,
        delta=1,
        thsad=(400, 300),
        limit=(2.0, 2.0),
        thscd=(400, 0.5),
        planes=[0],
    )
    assert denoised.format.id == clip.format.id
    assert denoised.num_frames == clip.num_frames


def test_mc_clamp() -> None:
    src = core.std.BlankClip(format=vs.GRAY8, width=160, height=120, length=5, color=10)
    flt = core.std.BlankClip(format=vs.GRAY8, width=160, height=120, length=5, color=20)

    mv = MVTools(src)
    mv.analyze(tr=1)

    # Clamp with zero margin (should clamp exactly to src value 10)
    clamped = mc_clamp(flt, src, mv, clamp=0)
    assert clamped.format.id == flt.format.id
    assert clamped.num_frames == flt.num_frames
    assert clamped.get_frame(0)[0][0, 0] == 10

    # Clamp with asymmetric margin (clamp undershoot=2, overshoot=3)
    # Since src is 10 and flt is 20, upper bound is 10 + 3 = 13.
    clamped_margin = mc_clamp(flt, src, mv, clamp=(2, 3))
    assert clamped_margin.get_frame(0)[0][0, 0] == 13
