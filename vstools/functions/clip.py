from __future__ import annotations

from ..exceptions import FramesLengthError
from ..types import FrameRange, VideoNodeT
from .normalize import normalize_franges

__all__ = [
    'shift_clip', 'shift_clip_multi',
]


def shift_clip(clip: VideoNodeT, offset: int) -> VideoNodeT:
    """
    Shift a clip forwards or backwards by *N* frames.

    This is useful for cases where you must compare every frame of a clip
    with the frame that comes before or after the current frame,
    like for example when performing temporal operations.

    Both positive and negative integers are allowed.
    Positive values will shift a clip forward, negative will shift a clip backward.

    :param clip:            Input clip.
    :param offset:          Number of frames to offset the clip with. Negative values are allowed.
                            Positive values will shift a clip forward,
                            negative will shift a clip backward.

    :return:                Clip that has been shifted forwards or backwards by *N* frames.
    """

    if offset > clip.num_frames - 1:
        raise FramesLengthError(shift_clip, 'offset')

    if offset < 0:
        return clip[0] * abs(offset) + clip[:offset]

    if offset > 0:
        return clip[offset:] + clip[-1] * offset

    return clip


def shift_clip_multi(clip: VideoNodeT, offsets: FrameRange = (-1, 1)) -> list[VideoNodeT]:
    """
    Shift a clip forwards or backwards multiple times by a varying amount of frames.

    This will return a clip for every shifting operation performed.
    This is a convenience function that makes handling multiple shifts easier.

    Example:

    .. code-block:: python

        >>> shift_clip_multi(clip, (-3, 3))
            [VideoNode, VideoNode, VideoNode, VideoNode, VideoNode, VideoNode, VideoNode]
                -3         -2         -1          0         +1         +2         +3

    :param clip:            Input clip.
    :param offsets:         List of frame ranges for offsetting.
                            A clip will be returned for every offset.
                            Default: (-1, 1).

    :return:                A list of clips, the amount determined by the amount of offsets.
    """

    ranges = normalize_franges(offsets)

    return [shift_clip(clip, x) for x in ranges]
