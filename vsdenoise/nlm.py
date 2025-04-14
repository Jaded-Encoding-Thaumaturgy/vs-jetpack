"""
This module implements a wrapper for non local means denoisers
"""

from __future__ import annotations

import warnings
from enum import auto
from typing import TYPE_CHECKING, Any, Callable, Literal, NamedTuple, Sequence, overload

from vstools import (
    CustomEnum, CustomIntEnum, CustomValueError, KwargsT, PlanesT, check_progressive,
    check_variable, core, join, normalize_planes, normalize_seq, to_arr, vs
)

__all__ = [
    'ChannelMode', 'DeviceType', 'NLMWeightMode',

    'nl_means'
]


class ChannelMode(CustomEnum):
    """Enum representing the NLMeans channel operation mode."""

    NO_PLANES = auto()
    """Don't process any planes."""

    ALL_PLANES = auto()
    """Process all planes."""

    LUMA = auto()
    """Only process luma in YUV/GRAY."""

    CHROMA = auto()
    """Only process chroma in YUV."""

    CHROMA_U = auto()
    """Only process chroma U plane in YUV."""

    CHROMA_V = auto()
    """Only process chroma V plane in YUV."""

    @classmethod
    def from_planes(cls, planes: PlanesT) -> ChannelMode:
        """
        Get :py:attr:`ChannelMode` from a traditional ``planes`` param.

        :param planes:  Sequence of planes to be processed.

        :return:        :py:attr:`ChannelMode` value.
        """

        if planes is None:
            return cls.ALL_PLANES

        planes = to_arr(planes)

        if planes == []:
            return cls.NO_PLANES

        if planes == [0]:
            return cls.LUMA

        if 0 not in planes:
            if planes == [1]:
                return cls.CHROMA_U
            elif planes == [2]:
                return cls.CHROMA_V

            return cls.CHROMA

        return cls.ALL_PLANES

    def to_planes(self) -> PlanesT:
        if self is ChannelMode.ALL_PLANES:
            return None

        if self is ChannelMode.LUMA:
            return [0]

        return [1, 2]


class DeviceTypeWithInfo(str):
    kwargs: KwargsT

    def __new__(cls, val: str, **kwargs: Any) -> DeviceTypeWithInfo:
        self = super().__new__(cls, val)
        self.kwargs = kwargs
        return self

    if TYPE_CHECKING:
        from .nlm import DeviceType

        @overload
        def __call__(  # type: ignore
            self: Literal[DeviceType.CUDA], *, device_id: int | None = None, num_streams: int | None = None
        ) -> DeviceType:
            ...

        @overload
        def __call__(  # type: ignore
            self: Literal[DeviceType.AUTO], *, device_id: int | None = None, **kwargs: Any
        ) -> DeviceType:
            ...

        @overload
        def __call__(  # type: ignore
            self: Literal[DeviceType.CPU] | Literal[DeviceType.GPU] | Literal[DeviceType.ACCELERATOR], *,
            device_id: int | None = None, ocl_x: int | None = None, ocl_y: int | None = None, ocl_r: int | None = None,
            info: int | None = None
        ) -> DeviceType:
            ...

        def __call__(self, **kwargs: Any) -> DeviceTypeWithInfo:
            "Add kwargs depending on the device you're going to use."
            ...
    else:
        def __call__(self, **kwargs: Any) -> DeviceTypeWithInfo:
            return DeviceTypeWithInfo(str(self), **kwargs)

    def NLMeans(
        self: DeviceTypeWithInfo | DeviceType, clip: vs.VideoNode,
        h: float | None = None, d: int | None = None, a: int | None = None, s: int | None = None,
        channels: str | None = None, wmode: int | None = None, wref: float | None = None,
        ref: vs.VideoNode | None = None, **kwargs: Any
    ) -> vs.VideoNode:
        assert check_progressive(clip, nl_means)

        if self == DeviceType.AUTO and hasattr(core, 'nlm_cuda'):
            self = DeviceType.CUDA

        if self == DeviceType.CUDA and not hasattr(core, 'nlm_cuda'):
            raise CustomValueError("You can't use cuda device type, you are missing the nlm_cuda plugin!")

        funcs = list[Callable[..., vs.VideoNode]]()

        if self == DeviceType.CUDA:
            funcs.append(core.proxied.nlm_cuda.NLMeans)
        else:
            funcs.extend([core.proxied.knlm.KNLMeansCL, core.proxied.nlm_ispc.NLMeans])

        exceptions = list[Exception]()

        for func in funcs:
            try:
                return func(clip, d, a, s, h, channels, wmode, wref, ref, **(self.kwargs | kwargs))
            except Exception as _e:
                exceptions.append(_e)

        for x in (True, False):
            for e in exceptions:
                if not isinstance(e, AttributeError):
                    if 'no compatible opencl' not in str(e):
                        if x:
                            continue
                    raise e from None

        raise next(iter(exceptions))


class DeviceType(DeviceTypeWithInfo, CustomEnum):
    """Enum representing available OpenCL device on which to run the plugin."""

    ACCELERATOR = 'accelerator'
    """Dedicated OpenCL accelerators."""

    CPU = 'cpu'
    """An OpenCL device that is the host processor."""

    GPU = 'gpu'
    """An OpenCL device that is a GPU."""

    CUDA = 'cuda'
    """Use a Cuda GPU."""

    AUTO = 'auto'
    """Automatically detect device. Priority is "cuda" -> "accelerator" -> "gpu" -> "cpu"."""

    if not TYPE_CHECKING:
        def __call__(self, **kwargs: Any) -> DeviceTypeWithInfo:
            return DeviceTypeWithInfo(str(self), **kwargs)


class NLMWeightMode(CustomIntEnum):
    WELSCH = 0
    """
    Welsch weighting function has a faster decay, but still assigns positive weights to dissimilar blocks.
    Original Non-local means denoising weighting function.
    """

    BISQUARE_LR = 1
    """
    Modified Bisquare weighting function to be less robust.
    """

    BISQUARE_THR = 2
    """
    Bisquare weighting function use a soft threshold to compare neighbourhoods.
    The weight is 0 as soon as a given threshold is exceeded.
    """

    BISQUARE_HR = 3
    """
    Modified Bisquare weighting function to be even more robust.
    """

    def __call__(self, weight_ref: float = 1.0) -> NLMWeightModeAndRef:
        """
        :param weight_ref:  Amount of original pixel to contribute to the filter output,
                            relative to the weight of the most similar pixel found.

        :return:            Config with weight mode and ref.
        """
        return NLMWeightModeAndRef(self, weight_ref)


class NLMWeightModeAndRef(NamedTuple):
    weight_mode: NLMWeightMode
    weight_ref: float


def nl_means(
    clip: vs.VideoNode, strength: float | Sequence[float] = 1.2,
    tr: int | Sequence[int] = 1, sr: int | Sequence[int] = 2, simr: int | Sequence[int] = 4,
    device_type: DeviceType = DeviceType.AUTO, ref: vs.VideoNode | None = None,
    wmode: NLMWeightMode | NLMWeightModeAndRef = NLMWeightMode.WELSCH, planes: PlanesT = None,
    **kwargs: Any
) -> vs.VideoNode:
    """
    Convenience wrapper for NLMeans implementations.

    :param clip:            Source clip.
    :param strength:        Controls the strength of the filtering.\n
                            Larger values will remove more noise.
    :param tr:              Temporal Radius. Temporal size = `(2 * tr + 1)`.\n
                            Sets the number of past and future frames to uses for denoising the current frame.\n
                            tr=0 uses 1 frame, while tr=1 uses 3 frames and so on.\n
                            Usually, larger values result in better denoising.
    :param sr:              Search Radius. Spatial size = `(2 * sr + 1)^2`.\n
                            Sets the radius of the search window.\n
                            sr=1 uses 9 pixel, while sr=2 uses 25 pixels and so on.\n
                            Usually, larger values result in better denoising.
    :param simr:            Similarity Radius. Similarity neighbourhood size = `(2 * simr + 1) ** 2`.\n
                            Sets the radius of the similarity neighbourhood window.\n
                            The impact on performance is low, therefore it depends on the nature of the noise.
    :param ref:             Reference clip to do weighting calculation.
    :param wmode:           Weighting function to use.
    :param planes:          Set the clip planes to be processed.
    :param device_type:     Set the device to use for processing. The fastest device will be used by default.
    :param kwargs:          Additional arguments passed to the plugin.

    :return:                Denoised clip.
    """

    assert check_variable(clip, nl_means)

    planes = normalize_planes(clip, planes)

    if planes == []:
        return clip

    nstrength, ntr, nsr, nsimr = to_arr(strength), to_arr(tr), to_arr(sr), to_arr(simr)

    wmoder, wref = wmode if isinstance(wmode, NLMWeightModeAndRef) else wmode()

    kwargs.update(ref=ref, wmode=wmoder.value, wref=wref)

    params = dict[str, list[float] | list[int]](strength=nstrength, tr=ntr, sr=nsr, simr=nsimr)

    def _nl_means(i: int, channels: str) -> vs.VideoNode:
        return device_type.NLMeans(clip, nstrength[i], ntr[i], nsr[i], nsimr[i], channels, **kwargs)

    if clip.format.color_family in {vs.GRAY, vs.RGB}:
        for doc, p in params.items():
            if len(set(p)) > 1:
                warnings.warn(
                    f'nl_means: only "{doc}" first value will be used since clip is {clip.format.color_family.name}',
                    UserWarning
                )

        return _nl_means(0, 'AUTO')

    if (
        all(len(p) < 2 for p in params.values())
        and clip.format.subsampling_w == clip.format.subsampling_h == 0
        and planes == [0, 1, 2]
    ):
        return _nl_means(0, 'YUV')

    nstrength, (ntr, nsr, nsimr) = normalize_seq(nstrength, 2), (normalize_seq(x, 2) for x in (ntr, nsr, nsimr))

    luma = _nl_means(0, 'Y') if 0 in planes else None
    chroma = _nl_means(1, 'UV') if 1 in planes or 2 in planes else None

    return join({None: clip, tuple(planes): chroma, 0: luma})
