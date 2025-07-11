from __future__ import annotations

from typing import Any

from vstools import (
    ConstantFormatVideoNode,
    FieldBased,
    FieldBasedT,
    FunctionUtil,
    InvalidFramerateError,
    VSFunctionKwArgs,
    VSFunctionNoArgs,
    core,
    find_prop_rfs,
    join,
    vs,
)

from .blending import deblend
from .enums import IVTCycles, VFMMode

__all__ = ["jivtc", "sivtc", "vdecimate", "vfm"]


def sivtc(
    clip: vs.VideoNode,
    pattern: int = 0,
    tff: FieldBasedT | bool | None = None,
    ivtc_cycle: IVTCycles = IVTCycles.CYCLE_10,
) -> ConstantFormatVideoNode:
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
    tff: FieldBasedT | bool | None = None,
    chroma_only: bool = True,
    postprocess: VSFunctionKwArgs[vs.VideoNode, vs.VideoNode] = deblend,
    postdecimate: IVTCycles | None = IVTCycles.CYCLE_05,
    ivtc_cycle: IVTCycles = IVTCycles.CYCLE_10,
    final_ivtc_cycle: IVTCycles = IVTCycles.CYCLE_08,
    **kwargs: Any,
) -> ConstantFormatVideoNode:
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

    InvalidFramerateError.check(jivtc, clip, (30000, 1001))

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
    tff: FieldBasedT | bool | None = None,
    mode: VFMMode = VFMMode.TWO_WAY_MATCH_THIRD_COMBED,
    postprocess: vs.VideoNode | VSFunctionNoArgs[vs.VideoNode, vs.VideoNode] | None = None,
    **kwargs: Any,
) -> ConstantFormatVideoNode:
    """
    Perform field matching using VFM.

    This function uses VIVTC's VFM plugin to detect and match pairs of fields in telecined content.

    You can pass a post-processing clip or function that will act on leftover combed frames.
    If you pass a clip, it will replace combed frames with that clip. If you pass a function,
    it will run that function on your input clip and replace combed frames with it.

    Example usage:

    ```py
        # Run vsaa.NNEDI3 on combed frames
        vfm(clip, postprocess=NNEDI3().deinterlace)
    ```

    Args:
        clip: Input clip to field matching telecine on.
        tff: Field order of the input clip. If None, it will be automatically detected.
        mode: VFM matching mode. For more information, see [VFMMode][vsdeinterlace.VFMMode]. Default:
            VFMMode.TWO_WAY_MATCH_THIRD_COMBED.
        postprocess: Optional function or clip to process combed frames. If a function is passed, it should take a clip
            as input and return a clip as output. If a clip is passed, it will be used as the postprocessed clip.
        **kwargs: Additional keyword arguments to pass to VFM. For a list of parameters, see the VIVTC documentation.

    Returns:
        Field matched clip with progressive frames.
    """

    func = FunctionUtil(clip, vfm, None, (vs.YUV, vs.GRAY), 8)

    tff = FieldBased.from_param_or_video(tff, clip, True, func.func).field

    vfm_kwargs = dict[str, Any](order=tff, mode=mode)

    if block := kwargs.pop("block", None):
        if isinstance(block, int):
            block = (block, block)

        vfm_kwargs.update(blockx=block[0], blocky=block[1])

    if (y := kwargs.pop("y", None)) and not isinstance(y, int):
        vfm_kwargs.update(y0=y[0], y1=y[1])

    if not kwargs.get("clip2") and func.work_clip.format != clip.format:
        vfm_kwargs.update(clip2=clip)

    fieldmatch = func.work_clip.vivtc.VFM(**(vfm_kwargs | kwargs))

    if postprocess:
        if callable(postprocess):
            postprocess = postprocess(kwargs.get("clip2", clip))

        fieldmatch = find_prop_rfs(fieldmatch, postprocess, "_Combed", "==", 1)

    return func.return_clip(fieldmatch)


def vdecimate(clip: vs.VideoNode, weight: float = 0.0, **kwargs: Any) -> ConstantFormatVideoNode:
    """
    Perform frame decimation using VDecimate.

    This function uses VIVTC's VDecimate plugin to remove duplicate frames from telecined content.
    It's recommended to use the vfm function before running this.

    Args:
        clip: Input clip to decimate.
        weight: Weight for frame blending. If > 0, blends duplicate frames before dropping one. Default: 0.0 (frames are
            dropped, not blended).
        **kwargs: Additional keyword arguments to pass to VDecimate. For a list of parameters, see the VIVTC
            documentation.

    Returns:
        Decimated clip with duplicate frames removed or blended.
    """

    func = FunctionUtil(clip, vdecimate, None, (vs.YUV, vs.GRAY), (8, 16))

    vdecimate_kwargs = dict[str, Any]()

    if block := kwargs.pop("block", None):
        if isinstance(block, int):
            block = (block, block)

        vdecimate_kwargs.update(blockx=block[0], blocky=block[1])

    if not kwargs.get("clip2") and func.work_clip.format != clip.format:
        vdecimate_kwargs.update(clip2=clip)

    dryrun = kwargs.pop("dryrun", False)

    if dryrun or weight:
        stats = func.work_clip.vivtc.VDecimate(dryrun=True, **(vdecimate_kwargs | kwargs))

        if dryrun:
            return func.return_clip(stats)

        clip = kwargs.pop("clip2", clip)

        avg = clip.std.AverageFrames(weights=[0, 1 - weight, weight])
        splice = find_prop_rfs(clip, avg, "VDecimateDrop", "==", 1, stats)
        vdecimate_kwargs.update(clip2=splice)

    decimate = func.work_clip.vivtc.VDecimate(**(vdecimate_kwargs | kwargs))

    return func.return_clip(decimate)
