from __future__ import annotations

from typing import Any, Iterable, Iterator, Literal, Sequence, overload

import vapoursynth as vs
from jetpytools import T, FuncExceptT, norm_display_name, norm_func_name, normalize_list_to_ranges, to_arr
from jetpytools import (
    flatten as jetp_flatten,
    invert_ranges as jetp_invert_ranges,
    normalize_range as normalize_franges,
    normalize_ranges as jetp_normalize_ranges,
    normalize_seq as jetp_normalize_seq
)

from ..types import ConstantFormatVideoNode, FrameRangeN, FrameRangesN, PlanesT, VideoNodeIterableT, VideoNodeT

__all__ = [
    'normalize_seq',
    'normalize_planes', 'invert_planes',
    'to_arr',
    'flatten', 'flatten_vnodes',
    'normalize_list_to_ranges',
    'normalize_franges',
    'normalize_ranges',
    'invert_ranges',
    'normalize_param_planes',
    'norm_func_name', 'norm_display_name'
]


@overload
def normalize_seq(val: T | Sequence[T], length: int = 3) -> list[T]:
    ...


@overload
def normalize_seq(val: Any, length: int = 3) -> list[Any]:
    ...


def normalize_seq(val: T | Sequence[T], length: int = 3) -> list[T]:
    """Normalize a sequence to the given length."""

    return jetp_normalize_seq(val, length)


def normalize_planes(clip: vs.VideoNode, planes: PlanesT = None) -> list[int]:
    """
    Normalize a sequence of planes.

    :param clip:        Input clip.
    :param planes:      Array of planes. If None, returns all planes of the input clip's format.
                        Default: None.

    :return:            Sorted list of planes.
    """

    assert clip.format

    if planes is None or planes == 4:
        planes = list(range(clip.format.num_planes))
    else:
        planes = to_arr(planes)

    return list(sorted(set(planes).intersection(range(clip.format.num_planes))))


def invert_planes(clip: vs.VideoNode, planes: PlanesT = None) -> list[int]:
    """
    Invert a sequence of planes.

    :param clip:        Input clip.
    :param planes:      Array of planes. If None, selects all planes of the input clip's format.

    :return:            Sorted inverted list of planes.
    """
    return sorted(set(normalize_planes(clip, None)) - set(normalize_planes(clip, planes)))


def normalize_param_planes(
    clip: vs.VideoNode, param: T | Sequence[T], planes: PlanesT, null: T, func: FuncExceptT | None = None
) -> list[T]:
    """
    Normalize a value or sequence to a list mapped to the clip's planes.

    For any plane not included in `planes`, the corresponding output value is set to `null`.

    :param clip:    The input clip whose format and number of planes will be used to determine mapping.
    :param param:   A single value or a sequence of values to normalize across the clip's planes.
    :param planes:  The planes to apply the values to. Other planes will receive `null`.
    :param null:    The default value to use for planes that are not included in `planes`.
    :param func:    Function returned for custom error handling.

    :return:        A list of length equal to the number of planes in the clip, with `param` values or `null`.
    """
    func = func or normalize_param_planes

    from .check import check_variable_format

    assert check_variable_format(clip, func)

    planes = normalize_planes(clip, planes)

    return [
        p if i in planes else null for i, p in enumerate(normalize_seq(param, clip.format.num_planes))
    ]


@overload
def flatten(items: Iterable[Iterable[T]]) -> Iterator[T]:
    ...


@overload
def flatten(items: Iterable[Any]) -> Iterator[Any]:
    ...


@overload
def flatten(items: Any) -> Iterator[Any]:
    ...


def flatten(items: Any) -> Iterator[Any]:
    """Flatten an array of values, clips and frames included."""

    if isinstance(items, (vs.RawNode, vs.RawFrame)):
        yield items
    else:
        yield from jetp_flatten(items)


@overload
def flatten_vnodes(
    *clips: VideoNodeIterableT[VideoNodeT], split_planes: Literal[False] = ...
) -> Sequence[VideoNodeT]:
    ...

@overload
def flatten_vnodes(
    *clips: VideoNodeIterableT[VideoNodeT], split_planes: Literal[True] = ...
) -> Sequence[ConstantFormatVideoNode]:
    ...


@overload
def flatten_vnodes(
    *clips: VideoNodeIterableT[VideoNodeT], split_planes: bool = ...
) -> Sequence[VideoNodeT]:
    ...


def flatten_vnodes(
    *clips: VideoNodeIterableT[VideoNodeT], split_planes: bool = False
) -> Sequence[vs.VideoNode]:
    """
    Flatten an array of VideoNodes.

    :param clips:           An array of clips to flatten into a list.
    :param split_planes:    Optionally split the VideoNodes into their individual planes as well.
                            Default: False.

    :return:                Flattened list of VideoNodes.
    """

    from .utils import split

    nodes = list[VideoNodeT](flatten(clips))

    if not split_planes:
        return nodes

    return sum(map(split, nodes), list[ConstantFormatVideoNode]())


def normalize_ranges(clip: vs.VideoNode, ranges: FrameRangeN | FrameRangesN) -> list[tuple[int, int]]:
    """
    Normalize ranges to a list of positive ranges.

    Frame ranges can include `None` and negative values.
    None will be converted to either 0 if it's the first value in a FrameRange,
    or the clip's length if it's the second item.
    Negative values will be subtracted from the clip's length.

    Examples:

    .. code-block:: python

        >>> clip.num_frames
        1000
        >>> normalize_ranges(clip, (None, None))
        [(0, 999)]
        >>> normalize_ranges(clip, (24, -24))
        [(24, 975)]
        >>> normalize_ranges(clip, [(24, 100), (80, 150)])
        [(24, 150)]


    :param clip:        Input clip.
    :param ranges:      Frame range or list of frame ranges.

    :return:            List of positive frame ranges.
    """

    return jetp_normalize_ranges(ranges, clip.num_frames)


def invert_ranges(
    clipa: vs.VideoNode, clipb: vs.VideoNode | None, ranges: FrameRangeN | FrameRangesN
) -> list[tuple[int, int]]:
    """
    Invert FrameRanges.

    Example:

    .. code-block:: python

        >>> franges = [(100, 200), 600, (1200, 2400)]
        >>> invert_ranges(core.std.BlankClip(length=10000), core.std.BlankClip(length=10000), franges)
        [(0, 99), (201, 599), (601, 1199), (2401, 9999)]

    :param clipa:          Original clip.
    :param clipb:          Replacement clip.
    :param ranges:         Ranges to replace clipa (original clip) with clipb (replacement clip).
                           These ranges will be inverted. For more info, see `replace_ranges`.

    :return:                A list of inverted frame ranges.
    """

    return jetp_invert_ranges(ranges, clipa.num_frames, None if clipb is None else clipb.num_frames)
