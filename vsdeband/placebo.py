from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from typing_extensions import deprecated

from vstools import (
    ConstantFormatVideoNode,
    CustomIntEnum,
    KwargsT,
    check_variable,
    fallback,
    inject_self,
    join,
    normalize_seq,
    split,
    vs,
)

from .abstract import Debander

__all__ = ["Placebo", "PlaceboDither"]


@deprecated('"PlaceboDither" is deprecated and will be removed in a future version', category=DeprecationWarning)
class PlaceboDither(CustomIntEnum):
    NONE = -1
    """
    No dithering.
    """

    BLUE_NOISE = 0
    """
    Dither with blue noise. Very high quality, but requires the use of a
    LUT. Warning: Computing a blue noise texture with a large size can be
    very slow, however this only needs to be performed once. Even so, using
    this with a `lut_size` greater than 6 is generally ill-advised. This is
    the preferred/default dither method.
    """

    DEFAULT = BLUE_NOISE

    ORDERED_LUT = 1
    """
    Dither with an ordered (bayer) dither matrix, using a LUT. Low quality,
    and since this also uses a LUT, there's generally no advantage to picking
    this instead of `BLUE_NOISE`. It's mainly there for testing.
    """

    ORDERED_FIXED = 2
    """
    The same as `ORDERED_LUT`, but uses fixed function math instead
    of a LUT. This is faster, but only supports a fixed dither matrix size
    of 16x16 (equal to a `lut_size` of 4). Requires GLSL 130+.
    """

    WHITE_NOISE = 3
    """
    Dither with white noise. This does not require a LUT and is fairly cheap
    to compute. Unlike the other modes it doesn't show any repeating
    patterns either spatially or temporally, but the downside is that this
    is visually fairly jarring due to the presence of low frequencies in the
    noise spectrum. Used as a fallback when the above methods are not
    available.
    """

    @property
    def placebo_args(self) -> KwargsT:
        """
        Get arguments you must pass to .placebo.Debander for this dither mode.
        """
        if self is PlaceboDither.NONE:
            return {"dither": False, "dither_algo": 0}
        return {"dither": True, "dither_algo": self.value}


@deprecated('"Placebo" is deprecated, use "placebo_deband" instead.', category=DeprecationWarning)
@dataclass
class Placebo(Debander):
    """
    Debander wrapper around libplacebo plugin's Deband function.
    """

    radius: float | None = None
    thr: float | list[float] | None = None
    grain: float | list[float] | None = None

    iterations: int | None = None

    dither: PlaceboDither | None = None

    @deprecated('"Placebo.deband" is deprecated, use "placebo_deband" instead.', category=DeprecationWarning)
    @inject_self
    def deband(  # type: ignore[override]
        self,
        clip: vs.VideoNode,
        radius: float = 16.0,
        thr: float | list[float] = 3.0,
        iterations: int = 4,
        grain: float | list[float] = 0.0,
        dither: PlaceboDither = PlaceboDither.DEFAULT,
    ) -> vs.VideoNode:
        """
        Main deband function, wrapper for `placebo.Deband <https:/github.com/Lypheo/vs-placebo#vs-placebo>`_

        Args:
            clip: Source clip
            radius: The debanding filter's initial radius. The radius increases linearly for each iteration. A higher
                radius will find more gradients, but a lower radius will smooth more aggressively.
            thr: The debanding filter's cut-off threshold. Higher numbers increase the debanding strength dramatically,
                but progressively diminish image details.
            iterations: The number of debanding steps to perform per sample. Each step reduces a bit more banding, but
                takes time to compute. Note that the strength of each step falls off very quickly, so high numbers (> 4)
                are practically useless.
            grain: Add some extra noise to the image. This significantly helps cover up remaining quantization
                artifacts. Higher numbers add more noise. Note: When debanding HDR sources, even a small amount of grain
                can result in a very big change to the brightness level. It's recommended to either scale this value
                down or disable it entirely for HDR.
            dither: Specify what kind of Placebo dithering will be used.

        Returns:
            Debanded clip
        """

        assert check_variable(clip, self.__class__.deband)

        radius = fallback(self.radius, radius)
        thr = normalize_seq(fallback(self.thr, thr))
        iterations = fallback(self.iterations, iterations)
        ngrain = normalize_seq(fallback(self.grain, grain))
        dither = fallback(self.dither, dither)

        def _placebo(clip: vs.VideoNode, thr: float, grain: Any, planes: Iterable[int]) -> ConstantFormatVideoNode:
            plane = 0
            for p in planes:
                plane |= pow(2, p)
            return clip.placebo.Deband(plane, iterations, thr, radius, grain * (1 << 5) * 0.8, **dither.placebo_args)

        set_grn = set(ngrain)

        if set_grn == {0} or clip.format.num_planes == 1:
            debs = [_placebo(p, t, ngrain[0], [0]) for p, t in zip(split(clip), thr)]

            if len(debs) == 1:
                return debs[0]

            return join(debs, clip.format.color_family)

        plane_map = {tuple(i for i in range(clip.format.num_planes) if ngrain[i] == x): x for x in set_grn - {0}}

        debanded = clip
        for planes, grain_val in plane_map.items():
            if len({thr[p] for p in planes}) == 1:
                debanded = _placebo(debanded, thr[planes[0]], grain_val, planes)
            else:
                for p in planes:
                    debanded = _placebo(debanded, thr[p], grain_val, planes)

        return debanded
