from __future__ import annotations

from typing import Any

from jetpytools import fallback, normalize_seq

from vstools import (
    FieldBased,
    FieldBasedLike,
    FramerateMismatchError,
    UnsupportedFramerateError,
    VSFunctionKwArgs,
    VSFunctionNoArgs,
    core,
    join,
    vs,
)

from .blending import deblend
from .enums import IVTCycles, VFMMode

__all__ = ["jivtc", "sivtc", "vdecimate", "vfm"]


def sivtc(
    clip: vs.VideoNode,
    pattern: int = 0,
    tff: FieldBasedLike | bool | None = None,
    ivtc_cycle: IVTCycles = IVTCycles.CYCLE_10,
) -> vs.VideoNode:
    """
    Simplest form of a fieldmatching function.

    This is essentially a stripped-down JIVTC offering JUST the basic fieldmatching and decimation part.
    As such, you may need to combine multiple instances if patterns change throughout the clip.

    Args:
        clip: Clip to process.
        pattern: First frame of any clean-combed-combed-clean-clean sequence.
        tff: Top-Field-First.

    Returns:
        IVTC'd clip.
    """

    tff = FieldBased.from_param_or_video(tff, clip, True, sivtc).is_tff

    ivtc = clip.std.SeparateFields(tff).std.DoubleWeave(tff)
    ivtc = ivtc_cycle.decimate(ivtc, pattern)

    return FieldBased.PROGRESSIVE.apply(ivtc)


def jivtc(
    clip: vs.VideoNode,
    pattern: int,
    tff: FieldBasedLike | bool | None = None,
    chroma_only: bool = True,
    postprocess: VSFunctionKwArgs = deblend,
    postdecimate: IVTCycles | None = IVTCycles.CYCLE_05,
    ivtc_cycle: IVTCycles = IVTCycles.CYCLE_10,
    final_ivtc_cycle: IVTCycles = IVTCycles.CYCLE_08,
    **kwargs: Any,
) -> vs.VideoNode:
    """
    This function should only be used when a normal ivtc or ivtc + bobber leaves chroma blend to every fourth frame.
    You can disable chroma_only to use it for luma as well, but it is not recommended.

    Args:
        clip: Clip to process. Has to be 60i.
        pattern: First frame of any clean-combed-combed-clean-clean sequence.
        tff: Set top field first (True) or bottom field first (False).
        chroma_only: Decide whether luma too will be processed.
        postprocess: Function to run after second decimation. Should be either a bobber or a deblender.
        postdecimate: If the postprocess function doesn't decimate itself, put True.

    Returns:
        Inverse Telecined clip.
    """

    tff = FieldBased.from_param_or_video(tff, clip, True, jivtc).is_tff

    UnsupportedFramerateError.check(clip, (30000, 1001), jivtc)

    ivtced = clip.std.SeparateFields(tff).std.DoubleWeave(tff)
    ivtced = ivtc_cycle.decimate(ivtced, pattern)

    pprocess = postprocess(clip if postdecimate else ivtced, **kwargs)

    if postdecimate:
        pprocess = postdecimate.decimate(pprocess, pattern)

    inter = core.std.Interleave([ivtced, pprocess])
    final = final_ivtc_cycle.decimate(inter, pattern)

    final = join(ivtced, final) if chroma_only else final

    return FieldBased.ensure_presence(final, FieldBased.PROGRESSIVE)


def vfm(
    clip: vs.VideoNode,
    tff: FieldBasedLike | bool | None = None,
    field: int = 2,
    mode: VFMMode = VFMMode.TWO_WAY_MATCH_THIRD_COMBED,
    mchroma: bool = True,
    cthresh: int = 9,
    mi: int = 80,
    chroma: bool = True,
    block: int | tuple[int, int] = 16,
    y: tuple[int, int] = (16, 16),
    scthresh: float = 12,
    micmatch: int = 1,
    micout: bool = False,
    clip2: vs.VideoNode | None = None,
    postprocess: vs.VideoNode | VSFunctionNoArgs | None = None,
) -> vs.VideoNode:
    """
    VFM is a field matching filter that recovers the original progressive frames
    from a telecined stream. VFM's output will contain duplicated frames, which
    is why it must be further processed by a decimation filter, like VDecimate.

    Usage Example:
        ```python
        # Run vsaa.NNEDI3 on leftover combed frames
        vfm(clip, postprocess=NNEDI3(double_rate=False).deinterlace)
        ```

    Args:
        clip: Input clip.
        tff: Sets the field order of the clip. Normally the field order is obtained from the `_FieldBased` frame
            property. This parameter is only used for those frames where the `_FieldBased` property has an invalid
            value or doesn't exist. If the field order is wrong, VFM's output will be visibly wrong in mode 0.
        field: Sets the field to match from. This is the field that VFM will take from the current frame in case of p
            or n matches. It is recommended to make this the same as the field order, unless you experience matching
            failures with that setting. In certain circumstances changing the field that is used to match from can have
            a large impact on matching performance. 0 and 1 will disregard the `_FieldBased` frame property. 2 and 3
            will adapt to the field order obtained from the `_FieldBased` property. Defaults to 2.
        mode: Sets the matching mode or strategy to use. Plain 2-way matching (option 0) is the safest of all the
            options in the sense that it won't risk creating jerkiness due to duplicate frames when possible, but if
            there are bad edits or blended fields it will end up outputting combed frames when a good match might
            actually exist. 3-way matching + trying the 4th/5th matches if all 3 of the original matches are detected as
            combed (option 5) is the most risky in terms of creating jerkiness, but will almost always find a good frame
            if there is one. The other settings (options 1, 2, 3, and 4) are all somewhere in between options 0 and 5 in
            terms of risking jerkiness and creating duplicate frames vs. finding good matches in sections with bad
            edits, orphaned fields, blended fields, etc. Note that the combed condition here is not the same as the
            `_Combed` frame property. Instead it's a combination of relative and absolute threshold comparisons and
            can still lead to the match being changed even when the `_Combed` flag is not set on the original frame.
            Defaults to VFMMode.TWO_WAY_MATCH_THIRD_COMBED.
        mchroma: Sets whether or not chroma is included during the match comparisons. In most cases it is recommended
            to leave this enabled. Only if your clip has bad chroma problems such as heavy rainbowing or other artifacts
            should you set this to false. Setting this to false could also be used to speed things up at the cost of
            some accuracy. Defaults to True.
        cthresh: This is the area combing threshold used for combed frame detection. This essentially controls how
            "strong" or "visible" combing must be to be detected. Larger values mean combing must be more visible and
            smaller values mean combing can be less visible or strong and still be detected. Valid settings are from -1
            (every pixel will be detected as combed) to 255 (no pixel will be detected as combed). This is basically a
            pixel difference value. A good range is between 8 to 12. Defaults to 9.
        mi: The number of combed pixels inside any of the `blockx` by `blocky` size blocks on the frame for the frame
            to be detected as combed. While `cthresh` controls how "visible" the combing must be, this setting controls
            "how much" combing there must be in any localized area (a window defined by the `blockx` and `blocky`
            settings) on the frame. The minimum is 0, the maximum is `blocky` * `blockx` (at which point no frames will
            ever be detected as combed). Defaults to 80.
        chroma: Sets whether or not chroma is considered in the combed frame decision. Only disable this if your source
            has chroma problems (rainbowing, etc) that are causing problems for the combed frame detection with `chroma`
            enabled. Actually, using chroma=false is usually more reliable, except in case there is chroma-only combing
            in the source. Defaults to True.
        block: Sets the size of the window used during combed frame detection. This has to do with the size of the area
            in which `mi` number of pixels are required to be detected as combed for a frame to be declared combed. See
            the `mi` parameter description for more info. Possible values are any power of 2 between 4 and 512. Defaults
            to 16.
        y: The rows from `y0` to `y1` will be excluded from the field matching decision. This can be used to ignore
            subtitles, a logo, or other things that may interfere with the matching. Set `y0` equal to `y1` to disable.
            Defaults to (16, 16).
        scthresh: Sets the scenechange threshold as a percentage of maximum change on the luma plane. Good values are
            in the 8 to 14 range. Defaults to 12.
        micmatch: When micmatch is greater than 0, tfm will take into account the mic values of matches when deciding
            what match to use as the final match. Only matches that could be used within the current matching mode are
            considered. micmatch has 3 possible settings:

               - 0: disabled. Modes 1, 2 and 3 effectively become identical to mode 0. Mode 5 becomes identical to mode
                4.
               - 1: micmatching will be used only around scene changes. See the `scthresh` parameter.
               - 2: micmatching will be used everywhere.

            Defaults to 1.
        micout: If true, VFM will calculate the mic values for all possible matches (p/c/n/b/u). Otherwise, only the
            mic values for the matches allowed by `mode` will be calculated. Defaults to False.
        clip2: Clip that VFM will use to create the output frames. If `clip2` is used, VFM will perform all
            calculations based on `clip`, but will copy the chosen fields from `clip2`. This can be used to work around
            VFM's video format limitations. Defaults to None.
        postprocess: Optional function or clip to process combed frames. If a function is passed, it should take a clip
            as input and return a clip as output. If a clip is passed, it will be used as the postprocessed clip. The
            output of the clip or function must have the same framerate as the input clip. Defaults to None.

    Returns:
        Field matched clip with progressive frames.
    """

    tff = FieldBased.from_param_or_video(tff, clip, True, vfm).is_tff

    nblock = normalize_seq(block, 2)

    if clip2 is None and clip.format not in (vs.YUV420P8, vs.YUV422P8, vs.YUV440P8, vs.YUV444P8, vs.GRAY8):
        new_family = vs.GRAY if clip.format.color_family is vs.GRAY else vs.YUV
        new_subsampling_w = min(clip.format.subsampling_w, 2)
        new_subsampling_h = min(clip.format.subsampling_h, 2)

        clip2 = clip
        clip = clip.resize.Bilinear(
            format=clip.format.replace(
                color_family=new_family,
                sample_type=vs.SampleType.INTEGER,
                bits_per_sample=8,
                subsampling_w=new_subsampling_w,
                subsampling_h=new_subsampling_h,
            )
        )

    fieldmatch = core.vivtc.VFM(
        clip,
        tff,
        field,
        mode,
        mchroma,
        cthresh,
        mi,
        chroma,
        nblock[0],
        nblock[1],
        y[0],
        y[1],
        scthresh,
        micmatch,
        micout,
        clip2,
    )

    if postprocess:
        if callable(postprocess):
            postprocess = postprocess(fallback(clip2, clip))

        FramerateMismatchError.check(
            vfm, clip, postprocess, message="The post-processed clip must be the same framerate as the input clip!"
        )

        fieldmatch = core.akarin.Select([fieldmatch, postprocess], fieldmatch, "x._Combed")

    return fieldmatch


def vdecimate(
    clip: vs.VideoNode,
    cycle: int = 5,
    chroma: bool = True,
    dupthresh: float = 1.1,
    scthresh: float = 15,
    block: int | tuple[int, int] = 16,
    clip2: vs.VideoNode | None = None,
    ovr: str | bytes | bytearray | None = None,
    dryrun: bool = False,
) -> vs.VideoNode:
    """
    VDecimate is a decimation filter. It drops one in every `cycle` frames - the one that is most likely to be a
    duplicate.

    Args:
        clip: Input clip.
        cycle: Size of a cycle, in frames. One in every `cycle` frames will be decimated. Defaults to 5.
        chroma: Controls whether the chroma is considered when calculating frame difference metrics. Defaults to True.
        dupthresh: This sets the threshold for duplicate detection. If the difference metric for a frame is less than
            or equal to this value then it is declared a duplicate. This value is a percentage of maximum change for a
            block defined by the `blockx` and `blocky` values, so 1.1 means 1.1% of maximum possible change. Defaults to
            1.1.
        scthresh: Sets the threshold for detecting scene changes. This value is a percentage of maximum change for the
            luma plane. Good values are between 10 and 15. Defaults to 15.
        block: Sets the size of the blocks used for metric calculations. Larger blocks give better noise suppression,
            but also give worse detection of small movements. Possible values are any power of 2 between 4 and 512.
            Defaults to 16.
        clip2: Clip that VDecimate will use to create the output frames. If `clip2` is used, VDecimate will perform all
            calculations based on `clip`, but will decimate frames from `clip2`. This can be used to work around
            VDecimate's video format limitations. Defaults to None.
        ovr: Text file containing overrides. This can be used to manually choose what frames get dropped.
            The frame numbers apply to the undecimated input clip, of course The decimation pattern must contain `cycle`
            characters If the overrides mark more than one frame per cycle, the first frame marked for decimation in the
            cycle will be dropped. Lines starting with # are ignored.

               - Drop a specific frame: 314 -
               - Drop every fourth frame, starting at frame 1001, up to frame 5403: 1001,5403 +++-

            Defaults to None.
        dryrun: If True, VDecimate will not drop any frames.
            Instead, it will attach the following properties to everyframe:

               - VDecimateDrop: 1 if VDecimate would normally drop the frame, 0 otherwise.
               - VDecimateMaxBlockDiff: This is the highest absolute difference between the current frame and the
                   previous frame found in any `blockx` `blocky` block.
               - VDecimateTotalDiff: This is the absolute difference between the current frame and the previous frame.

            Defaults to False.

    Returns:
         Decimated clip.
    """

    nblock = normalize_seq(block, 2)

    if clip2 is None and (clip.format.sample_type is not vs.SampleType.INTEGER or clip.format.bits_per_sample > 16):
        new_bits = min(clip.format.bits_per_sample, 16)

        clip2 = clip
        clip = clip.resize.Bilinear(
            format=clip.format.replace(sample_type=vs.SampleType.INTEGER, bits_per_sample=new_bits)
        )

    return core.vivtc.VDecimate(clip, cycle, chroma, dupthresh, scthresh, nblock[0], nblock[1], clip2, ovr, dryrun)
