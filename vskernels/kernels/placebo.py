from __future__ import annotations

from math import ceil
from typing import Any, ClassVar

from jetpytools import fallback, to_arr

from vskernels.types import BorderHandling, SampleGridModel
from vstools import UnsupportedVideoFormatError, core, get_video_format, vs
from vstools.enums.other import Dar, Sar

from ..abstract import ComplexScaler
from ..types import Center, LeftShift, Slope, TopShift

__all__ = [
    "EwaBicubic",
    "EwaGinseng",
    "EwaHann",
    "EwaJinc",
    "EwaLanczos",
    "EwaLanczos4Sharpest",
    "EwaLanczosSharp",
    "EwaRobidoux",
    "EwaRobidouxSharp",
    "Placebo",
]


class Placebo(ComplexScaler, abstract=True):
    """
    Abstract Placebo scaler class.

    This class and its subclasses depend on [vs-placebo](https://github.com/sgt0/vs-placebo).
    """

    _kernel: ClassVar[str]
    """Name of the Placebo kernel"""

    scale_function = core.lazy.placebo.Resample

    def __init__(
        self,
        radius: float | None = None,
        b: float | None = None,
        c: float | None = None,
        clamp: float = 0.0,
        blur: float = 0.0,
        taper: float = 0.0,
        antiring: float = 0.0,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the scaler with optional arguments.

        Keyword arguments passed during initialization are automatically injected here,
        unless explicitly overridden by the arguments provided at call time.
        Only arguments that match named parameters in this method are injected.

        Args:
            radius: Overrides the filter kernel radius. Has no effect if the filter kernel is not resizeable.
            b: The 'b' parameter for bicubic interpolation.
            c: The 'c' parameter for bicubic interpolation.
            clamp: Represents an extra weighting/clamping coefficient for negative weights. A value of 0.0 represents no
                clamping. A value of 1.0 represents full clamping, i.e. all negative lobes will be removed.
            blur: Additional blur coefficient. This effectively stretches the kernel, without changing the effective
                radius of the filter radius.
            taper: Additional taper coefficient. This essentially flattens the function's center.
            antiring: Antiringing strength.
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        self.radius = radius
        self.b = b
        self.c = c
        self.clamp = clamp
        self.blur = blur
        self.taper = taper
        self.antiring = antiring
        super().__init__(**kwargs)

    def scale(
        self,
        clip: vs.VideoNode,
        width: int | None = None,
        height: int | None = None,
        # ComplexScaler adds shift per planes
        shift: tuple[TopShift | list[TopShift], LeftShift | list[LeftShift]] = (0, 0),
        *,
        # `linear` and `sigmoid` from LinearScaler
        linear: bool | None = None,
        sigmoid: bool | tuple[Slope, Center] = False,
        # `border_handling`, `sample_grid_model`, `sar`, `dar`, `dar_in` and `keep_ar` from KeepArScaler
        border_handling: BorderHandling = BorderHandling.MIRROR,
        sample_grid_model: SampleGridModel = SampleGridModel.MATCH_EDGES,
        sar: Sar | float | bool | None = None,
        dar: Dar | float | bool | None = None,
        dar_in: Dar | bool | float | None = None,
        keep_ar: bool | None = None,
        # ComplexScaler adds blur
        blur: float | None = None,
        **kwargs: Any,
    ) -> vs.VideoNode:
        """
        Scale a clip to the given resolution, with aspect ratio and linear light support.

        Keyword arguments passed during initialization are automatically injected here,
        unless explicitly overridden by the arguments provided at call time.
        Only arguments that match named parameters in this method are injected.

        If a `format` is provided, only a change in chroma subsampling is allowed.
        Other format characteristics (bit depth, sample type, color family, etc.) must match the source clip.

        Args:
            clip: The source clip.
            width: Target width (defaults to clip width if None).
            height: Target height (defaults to clip height if None).
            shift: Subpixel shift (top, left) applied during scaling. If a tuple is provided, it is used uniformly. If a
                list is given, the shift is applied per plane.
            linear: Whether to linearize the input before descaling. If None, inferred from sigmoid.
            sigmoid: Whether to use sigmoid transfer curve. Can be True, False, or a tuple of (slope, center). `True`
                applies the defaults values (6.5, 0.75). Keep in mind sigmoid slope has to be in range 1.0-20.0
                (inclusive) and sigmoid center has to be in range 0.0-1.0 (inclusive).
            border_handling: Method for handling image borders during sampling.
            sample_grid_model: Model used to align sampling grid.
            sar: Sample aspect ratio to assume or convert to.
            dar: Desired display aspect ratio.
            dar_in: Input display aspect ratio, if different from clip's.
            keep_ar: Whether to adjust dimensions to preserve aspect ratio.
            blur: Amount of blur to apply during scaling.

        Returns:
            Scaled clip, optionally aspect-corrected and linearized.
        """
        if fmt := fallback(kwargs.get("format"), self.kwargs.get("format"), False):
            fmt = get_video_format(fmt)

            if (
                any(
                    in_v != out_v
                    for (k, in_v), out_v in zip(clip.format._as_dict().items(), fmt._as_dict().values())
                    if not k.startswith("subsampling")
                )
                and clip.format.color_family is fmt.color_family is vs.YUV
            ):
                raise UnsupportedVideoFormatError(
                    "Only YUV subsampling scaling is supported.", self.__class__, fmt.name
                )

            shift = (to_arr(shift[0]), to_arr(shift[1]))

        return super().scale(
            clip,
            width,
            height,
            shift,
            linear=linear,
            sigmoid=sigmoid,
            border_handling=border_handling,
            sample_grid_model=sample_grid_model,
            sar=sar,
            dar=dar,
            dar_in=dar_in,
            keep_ar=keep_ar,
            blur=blur,
            **kwargs,
        )

    def get_scale_args(
        self,
        clip: vs.VideoNode,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        width: int | None = None,
        height: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        kwargs.pop("format", None)

        return (
            {
                "sx": kwargs.pop("src_left", shift[1]),
                "sy": kwargs.pop("src_top", shift[0]),
                "width": width,
                "height": height,
                "filter": self._kernel,
                "radius": self.radius,
                "param1": self.b,
                "param2": self.c,
                "clamp": self.clamp,
                "taper": self.taper,
                "blur": self.blur,
                "antiring": self.antiring,
            }
            | self.kwargs
            | kwargs
            | {"linearize": False, "sigmoidize": False}
        )

    @ComplexScaler.cachedproperty
    def kernel_radius(self) -> int:
        if self.radius:
            return ceil(self.radius)

        if self.b or self.c:
            b, c = fallback(self.b, 0), fallback(self.c, 0.5)

            if (b, c) == (0, 0):
                return 1

        return 2


class EwaBicubic(Placebo):
    """
    Ewa Bicubic resizer.
    """

    _kernel = "ewa_robidoux"

    def __init__(self, b: float = 0.0, c: float = 0.5, radius: int | None = None, **kwargs: Any) -> None:
        """
        Initialize the scaler with specific 'b' and 'c' parameters and optional arguments.

        These keyword arguments are automatically forwarded to the
        [implemented_funcs][vskernels.BaseScaler.implemented_funcs] methods but only if the method explicitly accepts
        them as named parameters.
        If the same keyword is passed to both `__init__` and one of the
        [implemented_funcs][vskernels.BaseScaler.implemented_funcs], the one passed to `func` takes precedence.

        Args:
            b: The 'b' parameter for bicubic interpolation.
            c: The 'c' parameter for bicubic interpolation.
            radius: Overrides the filter kernel radius.
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        if radius is None:
            radius = 1 if (b, c) == (0, 0) else 2

        super().__init__(radius, b, c, **kwargs)


class EwaLanczos(Placebo):
    """
    Ewa Lanczos resizer.
    """

    _kernel = "ewa_lanczos"

    def __init__(self, radius: float = 3.2383154841662362076499, **kwargs: Any) -> None:
        """
        Initialize the kernel with a specific radius and optional keyword arguments.

        These keyword arguments are automatically forwarded to the
        [implemented_funcs][vskernels.BaseScaler.implemented_funcs] methods but only if the method explicitly accepts
        them as named parameters.
        If the same keyword is passed to both `__init__` and one of the
        [implemented_funcs][vskernels.BaseScaler.implemented_funcs], the one passed to `func` takes precedence.

        Args:
            radius: Overrides the filter kernel radius. Has no effect if the filter kernel is not resizeable.
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        super().__init__(radius, None, None, **kwargs)


class EwaLanczosSharp(Placebo):
    """Ewa Lanczos resizer."""

    _kernel = "ewa_lanczossharp"

    def __init__(
        self, radius: float = 3.2383154841662362076499, blur: float = 0.98125058372237073562493, **kwargs: Any
    ) -> None:
        """
        Initialize the kernel with a specific radius and optional keyword arguments.

        These keyword arguments are automatically forwarded to the
        [implemented_funcs][vskernels.BaseScaler.implemented_funcs] methods but only if the method explicitly accepts
        them as named parameters.
        If the same keyword is passed to both `__init__` and one of the
        [implemented_funcs][vskernels.BaseScaler.implemented_funcs], the one passed to `func` takes precedence.

        Args:
            radius: Overrides the filter kernel radius. Has no effect if the filter kernel is not resizeable.
            blur: Additional blur coefficient. This effectively stretches the kernel,
                without changing the effective radius of the filter radius.
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        super().__init__(radius, None, None, blur=blur, **kwargs)


class EwaLanczos4Sharpest(Placebo):
    """Ewa Lanczos resizer."""

    _kernel = "ewa_lanczos4sharpest"

    def __init__(
        self,
        radius: float = 4.2410628637960698819573,
        blur: float = 0.88451209326050047745788,
        antiring: float = 0.8,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the kernel with a specific radius and optional keyword arguments.

        These keyword arguments are automatically forwarded to the
        [implemented_funcs][vskernels.BaseScaler.implemented_funcs] methods but only if the method explicitly accepts
        them as named parameters.
        If the same keyword is passed to both `__init__` and one of the
        [implemented_funcs][vskernels.BaseScaler.implemented_funcs], the one passed to `func` takes precedence.

        Args:
            radius: Overrides the filter kernel radius. Has no effect if the filter kernel is not resizeable.
            blur: Additional blur coefficient. This effectively stretches the kernel,
                without changing the effective radius of the filter radius.
            antiring: Antiringing strength.
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        super().__init__(radius, None, None, blur=blur, antiring=antiring, **kwargs)


class EwaJinc(Placebo):
    """
    Ewa Jinc resizer.
    """

    _kernel = "ewa_jinc"

    def __init__(self, radius: float = 3.2383154841662362076499, **kwargs: Any) -> None:
        """
        Initialize the kernel with a specific radius and optional keyword arguments.

        These keyword arguments are automatically forwarded to the
        [implemented_funcs][vskernels.BaseScaler.implemented_funcs] methods but only if the method explicitly accepts
        them as named parameters.
        If the same keyword is passed to both `__init__` and one of the
        [implemented_funcs][vskernels.BaseScaler.implemented_funcs], the one passed to `func` takes precedence.

        Args:
            radius: Overrides the filter kernel radius. Has no effect if the filter kernel is not resizeable.
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        super().__init__(radius, None, None, **kwargs)


class EwaGinseng(Placebo):
    """
    Ewa Ginseng resizer.
    """

    _kernel = "ewa_ginseng"

    def __init__(self, radius: float = 3.2383154841662362076499, **kwargs: Any) -> None:
        """
        Initialize the kernel with a specific radius and optional keyword arguments.

        These keyword arguments are automatically forwarded to the
        [implemented_funcs][vskernels.BaseScaler.implemented_funcs] methods but only if the method explicitly accepts
        them as named parameters.
        If the same keyword is passed to both `__init__` and one of the
        [implemented_funcs][vskernels.BaseScaler.implemented_funcs], the one passed to `func` takes precedence.

        Args:
            radius: Overrides the filter kernel radius. Has no effect if the filter kernel is not resizeable.
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        super().__init__(radius, None, None, **kwargs)


class EwaHann(Placebo):
    """
    Ewa Hann resizer.
    """

    _kernel = "ewa_hann"

    def __init__(self, radius: float = 3.2383154841662362076499, **kwargs: Any) -> None:
        """
        Initialize the kernel with a specific radius and optional keyword arguments.

        These keyword arguments are automatically forwarded to the
        [implemented_funcs][vskernels.BaseScaler.implemented_funcs] methods but only if the method explicitly accepts
        them as named parameters.
        If the same keyword is passed to both `__init__` and one of the
        [implemented_funcs][vskernels.BaseScaler.implemented_funcs], the one passed to `func` takes precedence.

        Args:
            radius: Overrides the filter kernel radius. Has no effect if the filter kernel is not resizeable.
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        super().__init__(radius, None, None, **kwargs)


class EwaRobidoux(Placebo):
    """
    Ewa Robidoux resizer.
    """

    _kernel = "ewa_robidoux"

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the kernel with optional keyword arguments.

        These keyword arguments are automatically forwarded to the
        [implemented_funcs][vskernels.BaseScaler.implemented_funcs] methods but only if the method explicitly accepts
        them as named parameters.
        If the same keyword is passed to both `__init__` and one of the
        [implemented_funcs][vskernels.BaseScaler.implemented_funcs], the one passed to `func` takes precedence.

        Args:
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        super().__init__(None, None, None, **kwargs)


class EwaRobidouxSharp(Placebo):
    """
    Ewa Robidoux Sharp resizer.
    """

    _kernel = "ewa_robidouxsharp"

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the kernel with optional keyword arguments.

        These keyword arguments are automatically forwarded to the
        [implemented_funcs][vskernels.BaseScaler.implemented_funcs] methods but only if the method explicitly accepts
        them as named parameters.
        If the same keyword is passed to both `__init__` and one of the
        [implemented_funcs][vskernels.BaseScaler.implemented_funcs], the one passed to `func` takes precedence.

        Args:
            **kwargs: Keyword arguments that configure the internal scaling behavior.
        """
        super().__init__(None, None, None, **kwargs)
