from __future__ import annotations

from math import ceil
from typing import Any, Callable, ClassVar

from jetpytools import inject_kwargs_params
from vstools import core, fallback, inject_self, vs

from ..types import LeftShift, TopShift
from .complex import ComplexScaler

__all__ = [
    'Placebo',
    'EwaBicubic',
    'EwaJinc',
    'EwaLanczos',
    'EwaGinseng',
    'EwaHann',
    'EwaRobidoux',
    'EwaRobidouxSharp',
]


class Placebo(ComplexScaler, abstract=True):
    """
    Abstract Placebo scaler.

    Dependencies:

    * vs-placebo <https://github.com/sgt0/vs-placebo>`_
    """

    _kernel: ClassVar[str]
    """Name of the placebo kernel"""

    # Kernel settings
    taps: float | None
    b: float | None
    c: float | None

    # Filter settings
    clamp: float
    blur: float
    taper: float

    # Quality settings
    antiring: float

    scale_function = core.lazy.placebo.Resample

    def __init__(
        self,
        taps: float | None = None, b: float | None = None, c: float | None = None,
        clamp: float = 0.0, blur: float = 0.0, taper: float = 0.0,
        antiring: float = 0.0,
        **kwargs: Any
    ) -> None:
        self.taps = taps
        self.b = b
        self.c = c
        self.clamp = clamp
        self.blur = blur
        self.taper = taper
        self.antiring = antiring
        super().__init__(**kwargs)

    @inject_kwargs_params
    def get_scale_args(
        self, clip: vs.VideoNode, shift: tuple[TopShift, LeftShift] = (0, 0),
        width: int | None = None, height: int | None = None,
        *funcs: Callable[..., Any], **kwargs: Any
    ) -> dict[str, Any]:
        return (
            dict(sx=shift[1], sy=shift[0])
            | self.get_clean_kwargs(*funcs)
            | self.get_params_args(False, clip, width, height, **kwargs)
        )

    def get_params_args(
        self, is_descale: bool, clip: vs.VideoNode, width: int | None = None, height: int | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        return dict(
            width=width, height=height, filter=self._kernel,
            radius=self.taps, param1=self.b, param2=self.c,
            clamp=self.clamp, taper=self.taper, blur=self.blur,
            antiring=self.antiring,
        ) | kwargs

    @inject_self.cached.property
    def kernel_radius(self) -> int:
        from .bicubic import Bicubic

        if self.taps:
            return ceil(self.taps)

        if self.b or self.c:
            return Bicubic(fallback(self.b, 0), fallback(self.c, 0.5)).kernel_radius

        return 2


class EwaBicubic(Placebo):
    _kernel = 'ewa_robidoux'

    def __init__(self, b: float = 0.0, c: float = 0.5, radius: int | None = None, **kwargs: Any) -> None:
        radius = kwargs.pop('taps', radius)

        if radius is None:
            from .bicubic import Bicubic

            radius = Bicubic(b, c).kernel_radius

        super().__init__(radius, b, c, **kwargs)


class EwaLanczos(Placebo):
    _kernel = 'ewa_lanczos'

    def __init__(self, taps: float = 3.2383154841662362076499, **kwargs: Any) -> None:
        super().__init__(taps, None, None, **kwargs)


class EwaJinc(Placebo):
    _kernel = 'ewa_jinc'

    def __init__(self, taps: float = 3.2383154841662362076499, **kwargs: Any) -> None:
        super().__init__(taps, None, None, **kwargs)


class EwaGinseng(Placebo):
    _kernel = 'ewa_ginseng'

    def __init__(self, taps: float = 3.2383154841662362076499, **kwargs: Any) -> None:
        super().__init__(taps, None, None, **kwargs)


class EwaHann(Placebo):
    _kernel = 'ewa_hann'

    def __init__(self, taps: float = 3.2383154841662362076499, **kwargs: Any) -> None:
        super().__init__(taps, None, None, **kwargs)


class EwaRobidoux(Placebo):
    _kernel = 'ewa_robidoux'

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(None, None, None, **kwargs)


class EwaRobidouxSharp(Placebo):
    _kernel = 'ewa_robidouxsharp'

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(None, None, None, **kwargs)
