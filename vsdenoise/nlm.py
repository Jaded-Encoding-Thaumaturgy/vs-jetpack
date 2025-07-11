"""
This module implements a wrapper for non local means denoisers
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Generic, Sequence

from jetpytools import CustomRuntimeError, CustomStrEnum, P, R

from vstools import (
    ConstantFormatVideoNode,
    CustomIntEnum,
    PlanesT,
    check_variable,
    core,
    join,
    normalize_planes,
    normalize_seq,
    to_arr,
    vs,
)

__all__ = ["nl_means"]


class NLMeans(Generic[P, R]):
    """
    Class decorator that wraps the [nl_means][vsdenoise.nlm.nl_means] function
    and adds enumerations relevant to its implementation.

    It is not meant to be used directly.
    """

    def __init__(self, nl_means_func: Callable[P, R]) -> None:
        self._func = nl_means_func

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self._func(*args, **kwargs)

    class Backend(CustomStrEnum):
        """
        Enum representing available backends on which to run the plugin.
        """

        AUTO = "auto"
        """
        Automatically selects the best available backend.
        Priority: "cuda" -> "accelerator" -> "gpu" -> "cpu" -> "ispc".
        """

        ACCELERATOR = "accelerator"
        """
        Dedicated OpenCL accelerators.
        """

        GPU = "gpu"
        """
        An OpenCL device that is a GPU.
        """

        CPU = "cpu"
        """
        An OpenCL device that is the host processor.
        """

        ISPC = "ispc"
        """
        ISPC (CPU-based) implementation.
        """

        CUDA = "cuda"
        """
        CUDA (GPU-based) implementation.
        """

        def NLMeans(self, clip: vs.VideoNode, *args: Any, **kwargs: Any) -> ConstantFormatVideoNode:  # noqa: N802
            """
            Applies the Non-Local Means denoising filter using the plugin associated with the selected backend.

            Args:
                clip: Source clip.
                *args: Positional arguments passed to the selected plugin.
                **kwargs: Keyword arguments passed to the selected plugin.

            Raises:
                CustomRuntimeError: If the selected backend is not available or unsupported.

            Returns:
                Denoised clip.
            """

            if self == NLMeans.Backend.CUDA:
                return clip.nlm_cuda.NLMeans(*args, **kwargs)

            if self in [NLMeans.Backend.ACCELERATOR, NLMeans.Backend.GPU, NLMeans.Backend.CPU]:
                return clip.knlm.KNLMeansCL(*args, **kwargs | {"device_type": self.value})

            if self == NLMeans.Backend.ISPC:
                return clip.nlm_ispc.NLMeans(*args, **kwargs)

            # Fallback selection based on available plugins
            if hasattr(core, "nlm_cuda"):
                return NLMeans.Backend.CUDA.NLMeans(clip, *args, **kwargs)

            if hasattr(core, "knlm"):
                return clip.knlm.KNLMeansCL(*args, **kwargs | {"device_type": "auto"})

            if hasattr(core, "nlm_ispc"):
                return NLMeans.Backend.ISPC.NLMeans(clip, *args, **kwargs)

            raise CustomRuntimeError(
                "No compatible plugin found. Please install one from: "
                "https://github.com/AmusementClub/vs-nlm-cuda, https://github.com/AmusementClub/vs-nlm-ispc "
                "or https://github.com/Khanattila/KNLMeansCL"
            )

    class WeightMode(CustomIntEnum):
        """
        Enum of weighting modes for Non-Local Means (NLM) denoiser.
        """

        wref: float | None

        def __init__(self, value: int, wref: float | None = None) -> None:
            self._value_ = value
            self.wref = wref

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

        def __call__(self, wref: float | None = None) -> NLMeans.WeightMode:
            """
            Args:
                wref: Amount of original pixel to contribute to the filter output, relative to the weight of the most
                    similar pixel found.

            Returns:
                Config with weight mode and ref.
            """
            new_enum = CustomIntEnum(self.__class__.__name__, NLMeans.WeightMode.__members__)  # type: ignore
            member = getattr(new_enum, self.name)
            member.wref = wref
            return member


@NLMeans
def nl_means(
    clip: vs.VideoNode,
    h: float | Sequence[float] = 1.2,
    tr: int | Sequence[int] = 1,
    a: int | Sequence[int] = 2,
    s: int | Sequence[int] = 4,
    backend: NLMeans.Backend = NLMeans.Backend.AUTO,
    ref: vs.VideoNode | None = None,
    wmode: NLMeans.WeightMode = NLMeans.WeightMode.WELSCH,
    planes: PlanesT = None,
    **kwargs: Any,
) -> ConstantFormatVideoNode:
    """
    Convenience wrapper for NLMeans implementations.

    Filter description [here](https://github.com/Khanattila/KNLMeansCL/wiki/Filter-description).

    Example:
        ```py
        denoised = nl_means(clip, 0.4, backend=nl_means.Backend.CUDA, ...)
        ```

    Args:
        clip: Source clip.
        h: Controls the strength of the filtering. Larger values will remove more noise.
        tr: Temporal Radius. Temporal size = `(2 * d + 1)`. Sets the number of past and future frames to uses for
            denoising the current frame. d=0 uses 1 frame, while d=1 uses 3 frames and so on. Usually, larger values
            result in better denoising. Also known as the `d` parameter.
        a: Search Radius. Spatial size = `(2 * a + 1)^2`. Sets the radius of the search window. a=1 uses 9 pixel, while
            a=2 uses 25 pixels and so on. Usually, larger values result in better denoising.
        s: Similarity Radius. Similarity neighbourhood size = `(2 * s + 1) ** 2`. Sets the radius of the similarity
            neighbourhood window. The impact on performance is low, therefore it depends on the nature of the noise.
        backend: Set the backend to use for processing.
        ref: Reference clip to do weighting calculation. Also known as the `rclip` parameter.
        wmode: Weighting function to use.
        planes: Which planes to process.
        **kwargs: Additional arguments passed to the plugin.

    Returns:
        Denoised clip.
    """

    assert check_variable(clip, nl_means)

    planes = normalize_planes(clip, planes)

    if not planes:
        return clip

    params = dict[str, list[float] | list[int]](h=to_arr(h), d=to_arr(tr), a=to_arr(a), s=to_arr(s))

    # TODO: Remove legacy support for old arguments.
    for sargs, kargs in zip(["strength", "sr", "simr"], ["h", "a", "s"]):
        if sargs in kwargs:
            warnings.warn(f"nl_means: '{sargs}' argument is deprecated, use '{kargs}' instead", DeprecationWarning)
            params[kargs] = to_arr(kwargs.pop(sargs))

    def _nl_means(i: int, channels: str) -> ConstantFormatVideoNode:
        return backend.NLMeans(
            clip,
            **{k: p[i] for k, p in params.items()},
            **{"channels": channels, "rclip": ref, "wmode": wmode, "wref": wmode.wref} | kwargs,
        )

    if clip.format.color_family in {vs.GRAY, vs.RGB}:
        for doc, p in params.items():
            if len(set(p)) > 1:
                warnings.warn(
                    f'nl_means: only "{doc}" first value will be used since clip is {clip.format.color_family.name}',
                    UserWarning,
                )

        return _nl_means(0, "AUTO")

    if (
        all(len(p) < 2 for p in params.values())
        and clip.format.subsampling_w == clip.format.subsampling_h == 0
        and planes == [0, 1, 2]
    ):
        return _nl_means(0, "YUV")

    for k, p in params.items():
        params[k] = normalize_seq(p, 2)

    luma = _nl_means(0, "Y") if 0 in planes else None
    chroma = _nl_means(1, "UV") if 1 in planes or 2 in planes else None

    return join({None: clip, tuple(planes): chroma, 0: luma})
