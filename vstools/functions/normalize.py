from __future__ import annotations

from typing import Any, Iterable, Iterator, Literal, Sequence, overload

import vapoursynth as vs
from jetpytools import (
    FuncExceptT,
    SoftRange,
    StrictRange,
    T,
    fallback,
    norm_display_name,
    norm_func_name,
    to_arr,
)
from jetpytools import flatten as jetp_flatten
from jetpytools import invert_ranges as jetp_invert_ranges
from jetpytools import normalize_list_to_ranges as jetp_normalize_list_to_ranges
from jetpytools import normalize_range as jetp_normalize_range
from jetpytools import normalize_ranges as jetp_normalize_ranges
from jetpytools import normalize_ranges_to_list as jetp_normalize_ranges_to_list
from jetpytools import normalize_seq as jetp_normalize_seq

from ..types import ConstantFormatVideoNode, FrameRangeN, FrameRangesN, PlanesT, VideoNodeIterableT, VideoNodeT

__all__ = [
    "flatten",
    "flatten_vnodes",
    "invert_planes",
    "invert_ranges",
    "norm_display_name",
    "norm_func_name",
    "normalize_franges",
    "normalize_list_to_ranges",
    "normalize_param_planes",
    "normalize_planes",
    "normalize_ranges",
    "normalize_ranges_to_list",
    "normalize_seq",
    "to_arr",
]


@overload
def normalize_seq(val: T | Sequence[T], length: int = 3) -> list[T]: ...


@overload
def normalize_seq(val: Any, length: int = 3) -> list[Any]: ...


def normalize_seq(val: T | Sequence[T], length: int = 3) -> list[T]:
    """
    Normalize a sequence to the given length.
    """

    return jetp_normalize_seq(val, length)


def normalize_planes(clip: vs.VideoNode, planes: PlanesT = None) -> list[int]:
    """
    Normalize a sequence of planes.

    Args:
        clip: Input clip.
        planes: Array of planes. If None, returns all planes of the input clip's format. Default: None.

    Returns:
        Sorted list of planes.
    """

    assert clip.format

    planes = list(range(clip.format.num_planes)) if planes is None or planes == 4 else to_arr(planes)

    return sorted(set(planes).intersection(range(clip.format.num_planes)))


def invert_planes(clip: vs.VideoNode, planes: PlanesT = None) -> list[int]:
    """
    Invert a sequence of planes.

    Args:
        clip: Input clip.
        planes: Array of planes. If None, selects all planes of the input clip's format.

    Returns:
        Sorted inverted list of planes.
    """
    return sorted(set(normalize_planes(clip, None)) - set(normalize_planes(clip, planes)))


def normalize_param_planes(
    clip: vs.VideoNode, param: T | Sequence[T], planes: PlanesT, null: T, func: FuncExceptT | None = None
) -> list[T]:
    """
    Normalize a value or sequence to a list mapped to the clip's planes.

    For any plane not included in `planes`, the corresponding output value is set to `null`.

    Args:
        clip: The input clip whose format and number of planes will be used to determine mapping.
        param: A single value or a sequence of values to normalize across the clip's planes.
        planes: The planes to apply the values to. Other planes will receive `null`.
        null: The default value to use for planes that are not included in `planes`.
        func: Function returned for custom error handling.

    Returns:
        A list of length equal to the number of planes in the clip, with `param` values or `null`.
    """
    func = func or normalize_param_planes

    from .check import check_variable_format

    assert check_variable_format(clip, func)

    planes = normalize_planes(clip, planes)

    return [p if i in planes else null for i, p in enumerate(normalize_seq(param, clip.format.num_planes))]


@overload
def flatten(items: Iterable[Iterable[T]]) -> Iterator[T]: ...


@overload
def flatten(items: Iterable[Any]) -> Iterator[Any]: ...


@overload
def flatten(items: Any) -> Iterator[Any]: ...


def flatten(items: Any) -> Iterator[Any]:
    """
    Flatten an array of values, clips and frames included.
    """

    if isinstance(items, (vs.RawNode, vs.RawFrame)):
        yield items
    else:
        yield from jetp_flatten(items)


@overload
def flatten_vnodes(
    *clips: VideoNodeIterableT[VideoNodeT], split_planes: Literal[False] = ...
) -> Sequence[VideoNodeT]: ...


@overload
def flatten_vnodes(
    *clips: VideoNodeIterableT[VideoNodeT], split_planes: Literal[True] = ...
) -> Sequence[ConstantFormatVideoNode]: ...


@overload
def flatten_vnodes(*clips: VideoNodeIterableT[VideoNodeT], split_planes: bool = ...) -> Sequence[VideoNodeT]: ...


def flatten_vnodes(*clips: VideoNodeIterableT[VideoNodeT], split_planes: bool = False) -> Sequence[vs.VideoNode]:
    """
    Flatten an array of VideoNodes.

    Args:
        *clips: An array of clips to flatten into a list.
        split_planes: Optionally split the VideoNodes into their individual planes as well. Default: False.

    Returns:
        Flattened list of VideoNodes.
    """

    from .utils import split

    nodes = list[VideoNodeT](flatten(clips))

    if not split_planes:
        return nodes

    return sum(map(split, nodes), list[ConstantFormatVideoNode]())


def normalize_franges(ranges: SoftRange, /, exclusive: bool | None = None) -> Sequence[int]:
    """
    Normalize ranges represented by a tuple to an iterable of frame numbers.

    :param ranges:      Ranges to normalize.
    :param exclusive:   Whether to use exclusive (Python-style) ranges.
                        Defaults to False.

    :return:            List of positive frame ranges.
    """
    from ..utils import replace_ranges

    return jetp_normalize_range(ranges, fallback(exclusive, replace_ranges.exclusive, False))


def normalize_list_to_ranges(
    flist: Iterable[int], min_length: int = 0, exclusive: bool | None = None
) -> list[StrictRange]:
    from ..utils import replace_ranges

    return jetp_normalize_list_to_ranges(flist, min_length, fallback(exclusive, replace_ranges.exclusive, False))


def normalize_ranges(
    clip: vs.VideoNode, ranges: FrameRangeN | FrameRangesN, exclusive: bool | None = None
) -> list[tuple[int, int]]:
    """
    Normalize ranges to a list of positive ranges.

    Frame ranges can include `None` and negative values.
    None will be converted to either 0 if it's the first value in a FrameRange,
    or the clip's length if it's the second item.
    Negative values will be subtracted from the clip's length.

    Examples:

        >>> clip.num_frames
        1000
        >>> normalize_ranges(clip, (None, None))
        [(0, 999)]
        >>> normalize_ranges(clip, (24, -24))
        [(24, 975)]
        >>> normalize_ranges(clip, [(24, 100), (80, 150)])
        [(24, 150)]

    Args:
        clip: Input clip.
        ranges: Frame range or list of frame ranges.
        exclusive: Whether to use exclusive (Python-style) ranges. Defaults to False.

    Returns:
        List of positive frame ranges.
    """
    from ..utils import replace_ranges

    return jetp_normalize_ranges(ranges, clip.num_frames, fallback(exclusive, replace_ranges.exclusive, False))


def normalize_ranges_to_list(ranges: Iterable[SoftRange], exclusive: bool | None = None) -> list[int]:
    from ..utils import replace_ranges

    return jetp_normalize_ranges_to_list(ranges, fallback(exclusive, replace_ranges.exclusive, False))


def invert_ranges(
    clipa: vs.VideoNode, clipb: vs.VideoNode | None, ranges: FrameRangeN | FrameRangesN, exclusive: bool | None = None
) -> list[tuple[int, int]]:
    """
    Invert FrameRanges.

    Example:

        >>> franges = [(100, 200), 600, (1200, 2400)]
        >>> invert_ranges(core.std.BlankClip(length=10000), core.std.BlankClip(length=10000), franges)
        [(0, 99), (201, 599), (601, 1199), (2401, 9999)]

    Args:
        clipa: Original clip.
        clipb: Replacement clip.
        ranges: Ranges to replace clipa (original clip) with clipb (replacement clip). These ranges will be inverted.
            For more info, see `replace_ranges`.
        exclusive: Whether to use exclusive (Python-style) ranges. Defaults to False.

    Returns:
        A list of inverted frame ranges.
    """
    from ..utils import replace_ranges

    return jetp_invert_ranges(
        ranges,
        clipa.num_frames,
        None if clipb is None else clipb.num_frames,
        fallback(exclusive, replace_ranges.exclusive, False),
    )
