from typing import Any, Iterator, Literal, Self, overload

from ._enums import AudioChannels, ColorFamily, SampleType
from ._typings import _VideoFormatInfo


__all__ = [
    'VideoFormat', 'ChannelLayout'
]


class VideoFormat:
    id: int
    name: str
    color_family: ColorFamily
    sample_type: SampleType
    bits_per_sample: int
    bytes_per_sample: int
    subsampling_w: int
    subsampling_h: int
    num_planes: int

    def _as_dict(self) -> _VideoFormatInfo: ...

    def replace(
        self, *,
        color_family: ColorFamily | None = None,
        sample_type: SampleType | None = None,
        bits_per_sample: int | None = None,
        subsampling_w: int | None = None,
        subsampling_h: int | None = None
    ) -> Self: ...

    @overload
    def __eq__(self, other: Self) -> bool: ...

    @overload
    def __eq__(self, other: Any) -> Literal[False]: ...



class ChannelLayout(int):
    def __contains__(self, layout: AudioChannels) -> bool: ...

    def __iter__(self) -> Iterator[AudioChannels]: ...

    @overload
    def __eq__(self, other: ChannelLayout) -> bool: ...

    @overload
    def __eq__(self, other: Any) -> Literal[False]: ...

    def __len__(self) -> int: ...
