from __future__ import annotations

from typing import Any, overload

from jetpytools import CustomRuntimeError

from vsrgtools import MeanMode
from vstools import CustomValueError, FormatsMismatchError, PlanesT, VSFunctionKwArgs, VSFunctionNoArgs, vs

__all__ = [
    "frequency_merge",
]


@overload
def frequency_merge(
    *_clips: vs.VideoNode
    | tuple[
        vs.VideoNode, VSFunctionNoArgs[vs.VideoNode, vs.VideoNode] | VSFunctionKwArgs[vs.VideoNode, vs.VideoNode] | None
    ],
    mode_high: MeanMode | vs.VideoNode = MeanMode.LEHMER,
    mode_low: MeanMode | vs.VideoNode = MeanMode.ARITHMETIC,
    lowpass: None = None,
    planes: PlanesT = None,
    **kwargs: Any,
) -> vs.VideoNode: ...


@overload
def frequency_merge(
    *_clips: vs.VideoNode,
    mode_high: MeanMode | vs.VideoNode = MeanMode.LEHMER,
    mode_low: MeanMode | vs.VideoNode = MeanMode.ARITHMETIC,
    lowpass: VSFunctionNoArgs[vs.VideoNode, vs.VideoNode] | VSFunctionKwArgs[vs.VideoNode, vs.VideoNode],
    planes: PlanesT = None,
    **kwargs: Any,
) -> vs.VideoNode: ...


def frequency_merge(
    *_clips: vs.VideoNode
    | tuple[
        vs.VideoNode, VSFunctionNoArgs[vs.VideoNode, vs.VideoNode] | VSFunctionKwArgs[vs.VideoNode, vs.VideoNode] | None
    ],
    mode_high: MeanMode | vs.VideoNode = MeanMode.LEHMER,
    mode_low: MeanMode | vs.VideoNode = MeanMode.ARITHMETIC,
    lowpass: VSFunctionNoArgs[vs.VideoNode, vs.VideoNode] | VSFunctionKwArgs[vs.VideoNode, vs.VideoNode] | None = None,
    planes: PlanesT = None,
    **kwargs: Any,
) -> vs.VideoNode:
    """
    Merges the frequency components of the input clips.

    Example:

       - Replacing the high-frequency details of a Blu-ray source with the sharper components from a web stream:

         ``` py
         delowpass = frequency_merge(
             bluray_src,
             (web_src, lambda clip: Lanczos(taps=4).scale(clip, blur=4 / 3, force=True)),
             mode_high=web_src,
             mode_low=bluray_src,
         )
         ```

       - Lehmer merging 34 sources (stop doing that please)

         ``` py
         merged = frequency_merge(*clips, lowpass=lambda clip: DFTTest().denoise(clip, ...))
         ```

    Args:
        *_clips: A variable number of tuples, each containing a clip and an optional lowpass filter function. The
            lowpass function can either be a standard function or one that supports per-plane filtering.
        mode_high: The mean mode to use for the high frequency components or specifying the clip with the high frequency
            components. If a clip is passed, it must be one of the clips provided in `_clips`. Defaults to
            `MeanMode.LEHMER`.
        mode_low: The mean mode to use for the low frequency components or specifying the clip with the low frequency
            components. If a clip is passed, it must be one of the clips provided in `_clips`. Defaults to
            `MeanMode.ARITHMETIC`.
        lowpass: A global lowpass function applied to all provided clips. If set, ``_clips`` must consist solely of
            VideoNodes.
        planes: Planes to process. If specified, this will be passed to the lowpass functions. If None, it won't be
            included. Defaults to None.
        **kwargs: Additional keyword arguments passed to the lowpass functions.

    Returns:
        A clip representing the merged output of the frequency-separated clips.
    """

    clips = list[vs.VideoNode]()
    lowpass_funcs = list[
        VSFunctionNoArgs[vs.VideoNode, vs.VideoNode] | VSFunctionKwArgs[vs.VideoNode, vs.VideoNode] | None
    ]()

    for c in _clips:
        if isinstance(c, tuple):
            clip, func = c
        else:
            clip, func = c, lowpass

        clips.append(clip)
        lowpass_funcs.append(func)

    if not lowpass_funcs:
        raise CustomValueError("You must pass at least one lowpass filter!", frequency_merge)

    FormatsMismatchError.check(frequency_merge, *clips)

    blurred_clips = list[vs.VideoNode]()

    for clip, filt in zip(clips, lowpass_funcs):
        if not filt:
            blurred_clip = clip
        else:
            if planes is not None:
                kwargs.update(planes=planes)

            blurred_clip = filt(clip, **kwargs)

        blurred_clips.append(blurred_clip)

    if isinstance(mode_low, vs.VideoNode):
        try:
            low_freqs = blurred_clips[clips.index(mode_low)]
        except ValueError as e:
            raise CustomValueError(
                "Could not retrieve low-frequency clip: `mode_low` must be one of the clips provided in `_clips`.",
                frequency_merge,
            ) from e
    else:
        low_freqs = mode_low(blurred_clips, planes=planes, func=frequency_merge)

    diffed_clips = list[vs.VideoNode | None]()

    for clip, blurred in zip(clips, blurred_clips):
        diffed_clips.append(None if clip == blurred else clip.std.MakeDiff(blurred, planes))

    if isinstance(mode_high, vs.VideoNode):
        try:
            high_freqs = diffed_clips[clips.index(mode_high)]
        except ValueError as e:
            raise CustomValueError(
                "Could not retrieve high-frequency clip: `mode_high` must be one of the clips provided in `_clips`.",
                frequency_merge,
            ) from e

        if not high_freqs:
            raise CustomRuntimeError("Could not retrieve high-frequency clip!", frequency_merge)
    else:
        high_freqs = mode_high([clip for clip in diffed_clips if clip], planes=planes, func=frequency_merge)

    return low_freqs.std.MergeDiff(high_freqs, planes)
