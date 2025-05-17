from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from math import ceil, floor
from types import NoneType
from typing import Any, Callable, NamedTuple, Self, TypeAlias, overload

from vskernels import Scaler
from vstools import KwargsT, Resolution, get_w, mod2, vs

__all__ = [
    'scale_var_clip',

    'CropRel',
    'CropAbs',
    'ScalingArgs'
]


def scale_var_clip(
    clip: vs.VideoNode,
    scaler: Scaler | Callable[[Resolution], Scaler] | Callable[[tuple[int, int]], Scaler],
    width: int | Callable[[Resolution], int] | Callable[[tuple[int, int]], int] | None,
    height: int | Callable[[Resolution], int] | Callable[[tuple[int, int]], int],
    shift: tuple[float, float] | Callable[[tuple[int, int]], tuple[float, float]] = (0, 0),
    debug: bool = False
) -> vs.VideoNode:
    """
    Scale a variable clip to constant or variable resolution.

    :param clip:            Source clip.
    :param scaler:          A scaler instance or a callable that returns a scaler instance.
    :param width:           A width integer or a callable that returns the width. 
                            If None, it will be calculated from the height and the aspect ratio of the clip.
    :param height:          A height integer or a callable that returns the height.
    :param shift:           Optional top shift, left shift tuple or a callable that returns the shifts.
                            Defaults to no shift.
    :param debug:           If True, the `var_width` and `var_height` props will be added to the clip.
    :return:                Scaled clip.
    """
    _cached_clips = dict[str, vs.VideoNode]()

    no_accepts_var = list[Scaler]()

    def _eval_scale(f: vs.VideoFrame, n: int) -> vs.VideoNode:
        key = f'{f.width}_{f.height}'

        if key not in _cached_clips:
            res = Resolution(f.width, f.height)

            norm_scaler = scaler(res) if callable(scaler) else scaler
            norm_shift = shift(res) if callable(shift) else shift
            norm_height = height(res) if callable(height) else height

            if width is None:
                norm_width = get_w(norm_height, res.width / res.height)
            else:
                norm_width = width(res) if callable(width) else width

            part_scaler = partial(
                norm_scaler.scale, width=norm_width, height=norm_height, shift=norm_shift
            )

            scaled = clip
            if (scaled.width, scaled.height) != (norm_width, norm_height):
                if norm_scaler not in no_accepts_var:
                    try:
                        scaled = part_scaler(clip)
                    except BaseException:
                        no_accepts_var.append(norm_scaler)

                if norm_scaler in no_accepts_var:
                    const_clip = clip.resize.Point(res.width, res.height)

                    scaled = part_scaler(const_clip)

            if debug:
                scaled = scaled.std.SetFrameProps(var_width=res.width, var_height=res.height)

            _cached_clips[key] = scaled

        return _cached_clips[key]

    if callable(width) or callable(height):
        out_clip = clip
    else:
        out_clip = clip.std.BlankClip(width, height)

    return out_clip.std.FrameEval(_eval_scale, clip, clip)


LeftCrop: TypeAlias = int
RightCrop: TypeAlias = int
TopCrop: TypeAlias = int
BottomCrop: TypeAlias = int


class CropRel(NamedTuple):
    left: int = 0
    right: int = 0
    top: int = 0
    bottom: int = 0


class CropAbs(NamedTuple):
    width: int
    height: int
    left: int = 0
    top: int = 0

    def to_rel(self, base_clip: vs.VideoNode) -> CropRel:
        return CropRel(
            self.left,
            base_clip.width - self.width - self.left,
            self.top,
            base_clip.height - self.height - self.top
        )


@dataclass
class ScalingArgs:
    width: int
    height: int
    src_width: float
    src_height: float
    src_top: float
    src_left: float
    mode: str = 'hw'

    def _do(self) -> tuple[bool, bool]:
        return 'h' in self.mode.lower(), 'w' in self.mode.lower()

    def _up_rate(self, clip: vs.VideoNode | None = None) -> tuple[float, float]:
        if clip is None:
            return 1.0, 1.0

        do_h, do_w = self._do()

        return (
            (clip.height / self.height) if do_h else 1.0,
            (clip.width / self.width) if do_w else 1.0
        )

    def kwargs(self, clip_or_rate: vs.VideoNode | float | None = None, /) -> KwargsT:
        kwargs = dict[str, Any]()

        do_h, do_w = self._do()

        if isinstance(clip_or_rate, (vs.VideoNode, NoneType)):
            up_rate_h, up_rate_w = self._up_rate(clip_or_rate)
        else:
            up_rate_h, up_rate_w = clip_or_rate, clip_or_rate

        if do_h:
            kwargs.update(
                src_height=self.src_height * up_rate_h,
                src_top=self.src_top * up_rate_h
            )

        if do_w:
            kwargs.update(
                src_width=self.src_width * up_rate_w,
                src_left=self.src_left * up_rate_w
            )

        return kwargs

    @overload
    @classmethod
    def from_args(
        cls,
        base_clip: vs.VideoNode,
        height: int, width: int | None = None,
        *,
        src_top: float = ..., src_left: float = ...,
        mode: str = 'hw'
    ) -> Self:
        """
        Get (de)scaling arguments for integer scaling.

        :param base_clip:       Source clip.
        :param height:          Target (de)scaling height.
        :param width:           Target (de)scaling width.
                                If None, it will be calculated from the height and the aspect ratio of the base_clip.
        :param src_top:         Vertical offset.
        :param src_left:        Horizontal offset.
        :param mode:            Scaling mode:
                                - "w" means only the width is calculated.
                                - "h" means only the height is calculated.
                                - "hw or "wh" mean both width and height are calculated.
        :return:                ScalingArgs object suitable for scaling functions.
        """

    @overload
    @classmethod
    def from_args(
        cls,
        base_clip: vs.VideoNode,
        height: float, width: float | None = ...,
        base_height: int | None = ..., base_width: int | None = ...,
        src_top: float = ..., src_left: float = ...,
        crop: tuple[LeftCrop, RightCrop, TopCrop, BottomCrop] | CropRel | CropAbs = ...,
        mode: str = 'hw'
    ) -> Self:
        """
        Get (de)scaling arguments for fractional scaling.

        :param base_clip:       Source clip.
        :param height:          Target (de)scaling height. Casting to float will ensure fractional calculations.
        :param width:           Target (de)scaling width. Casting to float will ensure fractional calculations.
                                If None, it will be calculated from the height and the aspect ratio of the base_clip.
        :param base_height:     The height from which to contain the clip. If None, it will be calculated from the height.
        :param base_width:      The width from which to contain the clip. If None, it will be calculated from the width.
        :param src_top:         Vertical offset.
        :param src_left:        Horizontal offset.
        :param crop:            Tuple of cropping values, or relative/absolute crop specification.
        :param mode:            Scaling mode:
                                - "w" means only the width is calculated.
                                - "h" means only the height is calculated.
                                - "hw or "wh" mean both width and height are calculated.
        :return:                ScalingArgs object suitable for scaling functions.
        """

    @classmethod
    def from_args(
        cls,
        base_clip: vs.VideoNode,
        height: int | float, width: int | float | None = None,
        base_height: int | None = None, base_width: int | None = None,
        src_top: float = 0, src_left: float = 0,
        crop: tuple[LeftCrop, RightCrop, TopCrop, BottomCrop] | CropRel | CropAbs | None = None,
        mode: str = 'hw'
    ) -> Self:
        if crop:
            if isinstance(crop, CropAbs):
                crop = crop.to_rel(base_clip)
            elif isinstance(crop, CropRel):
                pass
            else:
                crop = CropRel(*crop)
        else:
            crop = CropRel()

        ratio_height = height / base_clip.height

        if width is None:
            if isinstance(height, int):
                width = get_w(height, base_clip, 2)
            else:
                width = ratio_height * base_clip.width
            
        ratio_width = width / base_clip.width

        if all([
            isinstance(height, int),
            isinstance(width, int),
            base_height is None,
            base_width is None,
            crop == (0, 0, 0, 0)
        ]):
            return cls(int(width), int(height), int(width), int(height), src_top, src_left, mode)

        if base_height is None:
            base_height = mod2(ceil(height))

        if base_width is None:
            base_width = mod2(ceil(width))

        margin_left = (base_width - width) / 2 + ratio_width * crop.left
        margin_right = (base_width - width) / 2 + ratio_width * crop.right
        cropped_width = base_width - floor(margin_left) - floor(margin_right)

        margin_top = (base_height - height) / 2 + ratio_height * crop.top
        margin_bottom = (base_height - height) / 2 + ratio_height * crop.bottom
        cropped_height = base_height - floor(margin_top) - floor(margin_bottom)

        if isinstance(width, int) and crop.left == crop.right == 0:
            cropped_src_width = float(cropped_width)
        else:
            cropped_src_width = ratio_width * (base_clip.width - crop.left - crop.right)

        cropped_src_left = margin_left - floor(margin_left) + src_left

        if isinstance(height, int) and crop.top == crop.bottom == 0:
            cropped_src_height = float(cropped_height)
        else:
            cropped_src_height = ratio_height * (base_clip.height - crop.top - crop.bottom)

        cropped_src_top = margin_top - floor(margin_top) + src_top

        return cls(
            cropped_width, cropped_height,
            cropped_src_width, cropped_src_height,
            cropped_src_top, cropped_src_left,
            mode
        )
