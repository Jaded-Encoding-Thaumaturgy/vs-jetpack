from __future__ import annotations

from typing import Any, Iterable

from vsrgtools import MeanMode
from vstools import (
    CustomValueError, GenericVSFunction, PlanesT, flatten_vnodes,
    normalize_planes, normalize_seq, vs, FormatsMismatchError
)

from .fft import DFTTest

__all__ = [
    'frequency_merge',
]


def frequency_merge(
    *_clips: vs.VideoNode | Iterable[vs.VideoNode],
    mode_high: MeanMode | vs.VideoNode = MeanMode.LEHMER, mode_low: MeanMode | vs.VideoNode = MeanMode.ARITHMETIC,
    lowpass: GenericVSFunction | list[GenericVSFunction] = DFTTest.denoise, planes: PlanesT = None, **kwargs: Any
) -> vs.VideoNode:
    """
    Merges the frequency components of the input clips.
    :param _clips:      The clips to merge.
    :param mode_high:   The mean mode to use for the high frequency components or
                        specifying the clip with the high frequency components.
    :param mode_low:    The mean mode to use for the low frequency components or
                        specifying the clip with the low frequency components.
    :param lowpass:     The lowpass filter to used to extract high frequency components.
                            Example: `lowpass = lambda i: vsrgtools.box_blur(i, passes=3)`
    :param planes:      The planes to process. If None, all planes will be processed.
    """

    clips = flatten_vnodes(_clips)
    n_clips = len(clips)

    planes = normalize_planes(clips[0], planes)

    if not lowpass:
        raise CustomValueError('You must pass at least one lowpass filter!', frequency_merge)

    FormatsMismatchError.check(frequency_merge, *clips)

    blurred_clips = []
    for clip, filt in zip(clips, normalize_seq(lowpass, n_clips)):
        try:
            blurred_clips.append(clip if not filt else filt(clip, planes=planes, **kwargs))
        except Exception:
            blurred_clips.append(clip if not filt else filt(clip, **kwargs))

    if isinstance(mode_low, vs.VideoNode):
        low_freqs = blurred_clips[clips.index(mode_low)]
    else:
        low_freqs = mode_low(blurred_clips, planes=planes, func=frequency_merge)

    diffed_clips = []
    for clip, blur in zip(clips, normalize_seq(blurred_clips, n_clips)):
        diffed_clips.append(None if clip == blur else clip.std.MakeDiff(blur, planes))

    if isinstance(mode_high, vs.VideoNode):
        high_freqs = diffed_clips[clips.index(mode_high)]
    else:
        high_freqs = mode_high([clip for clip in diffed_clips if clip], planes=planes, func=frequency_merge)

    return low_freqs.std.MergeDiff(high_freqs, planes)
