from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from fractions import Fraction
from math import floor
from types import TracebackType
from typing import Any, Literal, Self, overload

from jetpytools import MISSING, CustomValueError, MissingT, normalize_seq, to_arr
from jetpytools import flatten as jetp_flatten

from vsjetpack import deprecated

from ..enums import Align, Matrix
from ..types import Planes
from ..vs_proxy import core, vs
from .check import check_variable
from .info import get_subsampling
from .props import get_props
from .scale import get_lowest_values, get_neutral_values, get_peak_values

__all__ = [
    "Padder",
    "change_fps",
    "flatten",
    "invert_planes",
    "match_clip",
    "normalize_param_planes",
    "normalize_planes",
    "padder",
    "padder_ctx",
]


def change_fps(clip: vs.VideoNode, fps: Fraction) -> vs.VideoNode:
    """
    Convert the framerate of a clip.

    This is different from AssumeFPS as this will actively adjust
    the framerate of a clip, rather than simply set the framerate properties.

    Args:
        clip: Input clip.
        fps: Framerate to convert the clip to. Must be a Fraction.

    Returns:
        Clip with the framerate converted and frames adjusted as necessary.
    """

    src_num, src_den = clip.fps_num, clip.fps_den

    dest_num, dest_den = fps.as_integer_ratio()

    if (dest_num, dest_den) == (src_num, src_den):
        return clip

    factor = (dest_num / dest_den) * (src_den / src_num)

    new_fps_clip = clip.std.BlankClip(length=floor(clip.num_frames * factor), fpsnum=dest_num, fpsden=dest_den)

    return new_fps_clip.std.FrameEval(lambda n: clip[min(round(n / factor), clip.num_frames - 1)])


def match_clip(
    clip: vs.VideoNode,
    ref: vs.VideoNode,
    dimensions: bool = True,
    vformat: bool = True,
    matrices: bool = True,
    length: bool = False,
) -> vs.VideoNode:
    """
    Try to match the formats, dimensions, etc. of a reference clip to match the original clip.

    Args:
        clip: Original clip.
        ref: Reference clip.
        dimensions: Whether to adjust the dimensions of the reference clip to match the original clip. If True, uses
            resize.Bicubic to resize the image. Default: True.
        vformat: Whether to change the reference clip's format to match the original clip's. Default: True.
        matrices: Whether to adjust the Matrix, Transfer, and Primaries of the reference clip to match the original
            clip. Default: True.
        length: Whether to adjust the length of the reference clip to match the original clip.
    """
    assert check_variable(clip, match_clip)
    assert check_variable(ref, match_clip)

    if length:
        if clip.num_frames < ref.num_frames:
            clip = vs.core.std.Splice([clip, clip[-1] * (ref.num_frames - clip.num_frames)])
        else:
            clip = clip[: ref.num_frames]

    clip = clip.resize.Bicubic(ref.width, ref.height) if dimensions else clip

    if vformat:
        clip = clip.resize.Bicubic(format=ref.format, matrix=Matrix.from_video(ref))

    if matrices:
        clip = clip.std.SetFrameProps(
            **get_props(ref, ["_Matrix", "_Transfer", "_Primaries"], int, default=2, func=match_clip)
        )

    return clip.std.AssumeFPS(fpsnum=ref.fps.numerator, fpsden=ref.fps.denominator)


@dataclass
class Padder(Sequence[int]):
    """Pad out the pixels on the sides of a clip."""

    left: int = 0
    """Padding added to the left side of the clip."""

    right: int = 0
    """Padding added to the right side of the clip."""

    top: int = 0
    """Padding added to the top side of the clip."""

    bottom: int = 0
    """Padding added to the bottom side of the clip."""

    input_width: int | None = None
    """Original width of the input clip."""

    input_height: int | None = None
    """Original height of the input clip."""

    @overload
    def __getitem__(self, i: int, /) -> int: ...
    @overload
    def __getitem__(self, s: slice[int | None, int | None, int | None], /) -> Sequence[int]: ...
    def __getitem__(self, value: Any) -> Any:
        return [self.left, self.right, self.top, self.bottom][value]

    def __len__(self) -> Literal[4]:
        return 4

    @property
    def padded_width(self) -> int:
        """Padded width of the clip."""
        if self.input_width is None:
            raise CustomValueError("Input width is not set. Specify input_width when instantiating Padder.", "Padder")
        return self.input_width + self.left + self.right

    @property
    def padded_height(self) -> int:
        """Padded height of the clip."""
        if self.input_height is None:
            raise CustomValueError("Input height is not set. Specify input_height when instantiating Padder.", "Padder")
        return self.input_height + self.top + self.bottom

    def crop(self, clip: vs.VideoNode, crop_scale: float | tuple[float, float] | None = None) -> vs.VideoNode:
        """
        Crop a clip with the padding values.

        Args:
            clip: Input clip.
            crop_scale: Optional scale factor for the padding values. If None, no scaling is applied.

        Returns:
            Cropped clip.
        """
        if crop_scale is None:
            crop_scale = (clip.width / self.padded_width, clip.height / self.padded_height)
        elif not isinstance(crop_scale, tuple):
            crop_scale = (crop_scale, crop_scale)

        left_pad, right_pad, top_pad, bottom_pad = tuple(crop_scale[0 if i < 2 else 1] for i in range(4))
        return clip.std.Crop(
            left=int(self.left * left_pad),
            right=int(self.right * right_pad),
            top=int(self.top * top_pad),
            bottom=int(self.bottom * bottom_pad),
        )

    def mirror(self, clip: vs.VideoNode) -> vs.VideoNode:
        """
        Pad a clip with reflect mode. This will reflect the clip on each side.

        Visual example:
            ```
            >>> |ABCDE
            >>> padder.MIRROR(left=3)
            >>> CBA|ABCDE
            ```

        Args:
            clip: Input clip.

        Returns:
            Padded clip with reflected borders.
        """
        width, height, *_ = self._get_values(clip)

        return (
            clip.std.RemoveFrameProps()
            .resize.Point(
                width,
                height,
                src_top=-self.top,
                src_left=-self.left,
                src_width=width,
                src_height=height,
            )
            .std.CopyFrameProps(clip)
        )

    def repeat(self, clip: vs.VideoNode) -> vs.VideoNode:
        """
        Pad a clip with repeat mode. This will simply repeat the last row/column till the end.

        Visual example:
            ```
            >>> |ABCDE
            >>> padder.REPEAT(left=3)
            >>> AAA|ABCDE
            ```

        Args:
            clip: Input clip.

        Returns:
            Padded clip with repeated borders.
        """
        *_, fmt, w_sub, h_sub = self._get_values(clip)

        padded = clip.std.AddBorders(self.left, self.right, self.top, self.bottom)

        right_idx, bottom_idx = clip.width + self.left, clip.height + self.top

        pads = [
            (self.left, right_idx, self.top, bottom_idx),
            (self.left // w_sub, right_idx // w_sub, self.top // h_sub, bottom_idx // h_sub),
        ][: fmt.num_planes]

        return padded.akarin.Expr(
            [
                """
                X {left} < L! Y {top} < T! X {right} > R! Y {bottom} > B!

                T@ B@ or L@ R@ or and
                    L@
                        T@ {left} {top}  x[] {left} {bottom} x[] ?
                        T@ {right} {top} x[] {right} {bottom} x[] ?
                    ?
                    L@
                        {left} Y x[]
                        R@
                            {right} Y x[]
                            T@
                                X {top} x[]
                                B@
                                    X {bottom} x[]
                                    x
                                ?
                            ?
                        ?
                    ?
                ?
            """.format(left=l_, right=r_ - 1, top=t_, bottom=b_ - 1)
                for l_, r_, t_, b_ in pads
            ]
        )

    def color(
        self,
        clip: vs.VideoNode,
        color: float | None | MissingT | Sequence[float | None | MissingT] = (False, MISSING),
    ) -> vs.VideoNode:
        """
        Pad a clip with a constant color.

        Visual example:
            ```
            >>> |ABCDE
            >>> padder.COLOR(left=3, color=Z)
            >>> ZZZ|ABCDE
            ```

        Args:
            clip: Input clip.
            color: Constant color that should be added on the sides:

                   * number: This will be treated as such and not converted or clamped.
                   * False: Lowest value for this clip format and color range.
                   * True: Highest value for this clip format and color range.
                   * None: Neutral value for this clip format.
                   * MISSING: Automatically set to False if RGB, else None.

        Returns:
            Padded clip with colored borders.
        """
        self._get_values(clip)

        def _norm(colr: float | bool | None | MissingT) -> Sequence[float]:
            if colr is MISSING:
                colr = False if clip.format.color_family is vs.RGB else None

            if colr is False:
                return get_lowest_values(clip, clip)

            if colr is True:
                return get_peak_values(clip, clip)

            if colr is None:
                return get_neutral_values(clip)

            return normalize_seq(colr, clip.format.num_planes)

        if not isinstance(color, Sequence):
            norm_colors = _norm(color)
        else:
            norm_colors = [_norm(c)[i] for i, c in enumerate(normalize_seq(color, clip.format.num_planes))]

        return core.std.AddBorders(clip, self.left, self.right, self.top, self.bottom, norm_colors)

    @classmethod
    def from_mod(cls, clip: vs.VideoNode, mod: int = 16, min: int = 4, align: Align = Align.MIDDLE_CENTER) -> Self:
        """
        Initialize Padder based on modulus constraints of a clip.

        Args:
            clip: Input clip to calculate mod padding for.
            mod: Modulus value (e.g. 16).
            min: Minimum padding required.
            align: Align option.

        Returns:
            Padder instance with calculated padding values and input clip size.
        """
        return cls(*cls.mod_padding(clip, mod, min, align), clip.width, clip.height)

    @classmethod
    def mod_padding(
        cls,
        sizes: tuple[int, int] | vs.VideoNode,
        mod: int = 16,
        min: int = 4,
        align: Align = Align.MIDDLE_CENTER,
    ) -> tuple[int, int, int, int]:
        """
        Calculate required padding to make clip dimensions divisible by modulus.

        Args:
            sizes: Width and height tuple or a VideoNode.
            mod: Modulus value (e.g. 16).
            min: Minimum padding required.
            align: Align option.

        Returns:
            A tuple of left, right, top, bottom padding values.
        """
        if isinstance(sizes, vs.VideoNode):
            sizes = (sizes.width, sizes.height)

        ph, pv = (mod - (((x + min * 2) - 1) % mod + 1) for x in sizes)
        left, top = floor(ph / 2), floor(pv / 2)
        left, right, top, bottom = tuple(x + min for x in (left, ph - left, top, pv - top))

        if align & Align.TOP:
            bottom += top
            top = 0
        elif align & Align.BOTTOM:
            top += bottom
            bottom = 0

        if align & Align.LEFT:
            right += left
            left = 0
        elif align & Align.RIGHT:
            left += right
            right = 0

        return left, right, top, bottom

    @classmethod
    def mod_padding_crop(
        cls,
        sizes: tuple[int, int] | vs.VideoNode,
        mod: int = 16,
        min: int = 4,
        crop_scale: float | tuple[float, float] = 2,
        align: Align = Align.MIDDLE_CENTER,
    ) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
        """
        Calculate mod padding and its corresponding cropped padding.

        Args:
            sizes: Width and height tuple or a VideoNode.
            mod: Modulus value (e.g. 16).
            min: Minimum padding required.
            crop_scale: Crop scale factor.
            align: Align option.

        Returns:
            A tuple containing padding tuple and scaled crop padding tuple.
        """
        if not isinstance(crop_scale, tuple):
            crop_scale = (crop_scale, crop_scale)
        padding = cls.mod_padding(sizes, mod, min, align)
        return padding, tuple(int(pad * crop_scale[0 if i < 2 else 1]) for i, pad in enumerate(padding))  # type: ignore[return-value]

    def _get_values(self, clip: vs.VideoNode) -> tuple[int, int, vs.VideoFormat, int, int]:
        width = clip.width + self.left + self.right
        height = clip.height + self.top + self.bottom

        w_sub, h_sub = 2**clip.format.subsampling_w, 2**clip.format.subsampling_h

        if width % w_sub or height % h_sub:
            raise CustomValueError(
                "Values must result in a mod congruent to the clip's subsampling ({subsampling}).",
                "Padder",
                subsampling=get_subsampling(clip),
            )

        return width, height, clip.format, w_sub, h_sub


# ruff: noqa: N802, N801
@deprecated("Use Padder instead", category=DeprecationWarning)
class padder_ctx(AbstractContextManager["padder_ctx"]):  # pyright: ignore[reportDeprecated]
    """
    Context manager for the padder class.
    """

    def __init__(self, mod: int = 8, min: int = 0, align: Align = Align.MIDDLE_CENTER) -> None:
        """
        Initializes the class

        Args:
            mod: The modulus used for calculations or constraints. Defaults to 8.
            min: The minimum value allowed or used as a base threshold. Defaults to 0.
            align: The alignment configuration. Defaults to Align.MIDDLE_CENTER.
        """
        self.mod = mod
        self.min = min
        self.align = align
        self.pad_ops = list[tuple[tuple[int, int, int, int], tuple[int, int]]]()

    def CROP(self, clip: vs.VideoNode, crop_scale: float | tuple[float, float] | None = None) -> vs.VideoNode:
        """
        Crop a clip with the padding values.

        Args:
            clip: Input clip.
            crop_scale: Optional scale factor for the padding values. If None, no scaling is applied.

        Returns:
            Cropped clip.
        """
        (padding, sizes) = self.pad_ops.pop(0)

        input_width = sizes[0] - padding[0] - padding[1]
        input_height = sizes[1] - padding[2] - padding[3]
        pad = Padder(padding[0], padding[1], padding[2], padding[3], input_width, input_height)
        return pad.crop(clip, crop_scale)

    def MIRROR(self, clip: vs.VideoNode) -> vs.VideoNode:
        """
        Pad a clip with reflect mode. This will reflect the clip on each side.

        Args:
            clip: Input clip.

        Returns:
            Padded clip with reflected borders.
        """
        padding = Padder.mod_padding(clip, self.mod, self.min, self.align)
        pad = Padder(*padding, clip.width, clip.height)
        out = pad.mirror(clip)
        self.pad_ops.append((padding, (out.width, out.height)))
        return out

    def REPEAT(self, clip: vs.VideoNode) -> vs.VideoNode:
        """
        Pad a clip with repeat mode. This will simply repeat the last row/column till the end.

        Args:
            clip: Input clip.

        Returns:
            Padded clip with repeated borders.
        """
        padding = Padder.mod_padding(clip, self.mod, self.min, self.align)
        pad = Padder(*padding, clip.width, clip.height)
        out = pad.repeat(clip)
        self.pad_ops.append((padding, (out.width, out.height)))
        return out

    def COLOR(self, clip: vs.VideoNode, color: float | None | Sequence[float | None] = (False, None)) -> vs.VideoNode:
        """
        Pad a clip with a constant color.

        Args:
            clip: Input clip.
            color: Constant color that should be added on the sides:

                   * number: This will be treated as such and not converted or clamped.
                   * False: Lowest value for this clip format and color range.
                   * True: Highest value for this clip format and color range.
                   * None: Neutral value for this clip format.
                   * MISSING: Automatically set to False if RGB, else None.

        Returns:
            Padded clip with colored borders.
        """
        padding = Padder.mod_padding(clip, self.mod, self.min, self.align)
        pad = Padder(*padding, clip.width, clip.height)
        out = pad.color(clip, color)
        self.pad_ops.append((padding, (out.width, out.height)))
        return out

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        return None


@deprecated("Use Padder instead", category=DeprecationWarning)
class padder:
    """
    Pad out the pixels on the sides by the given amount of pixels.
    """

    ctx = padder_ctx  # pyright: ignore[reportDeprecated]
    """Context manager for the padder class."""

    @staticmethod
    def MIRROR(clip: vs.VideoNode, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0) -> vs.VideoNode:
        """
        Pad a clip with reflect mode. This will reflect the clip on each side.

        Visual example:
            ```
            >>> |ABCDE
            >>> padder.MIRROR(left=3)
            >>> CBA|ABCDE
            ```

        Args:
            clip: Input clip.
            left: Padding added to the left side of the clip.
            right: Padding added to the right side of the clip.
            top: Padding added to the top side of the clip.
            bottom: Padding added to the bottom side of the clip.

        Returns:
            Padded clip with reflected borders.
        """
        return Padder(left, right, top, bottom).mirror(clip)

    @staticmethod
    def REPEAT(clip: vs.VideoNode, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0) -> vs.VideoNode:
        """
        Pad a clip with repeat mode. This will simply repeat the last row/column till the end.

        Visual example:
            ```
            >>> |ABCDE
            >>> padder.REPEAT(left=3)
            >>> AAA|ABCDE
            ```

        Args:
            clip: Input clip.
            left: Padding added to the left side of the clip.
            right: Padding added to the right side of the clip.
            top: Padding added to the top side of the clip.
            bottom: Padding added to the bottom side of the clip.

        Returns:
            Padded clip with repeated borders.
        """
        return Padder(left, right, top, bottom).repeat(clip)

    @staticmethod
    def COLOR(
        clip: vs.VideoNode,
        left: int = 0,
        right: int = 0,
        top: int = 0,
        bottom: int = 0,
        color: float | None | MissingT | Sequence[float | None | MissingT] = (False, MISSING),
    ) -> vs.VideoNode:
        """
        Pad a clip with a constant color.

        Visual example:
            ```
            >>> |ABCDE
            >>> padder.COLOR(left=3, color=Z)
            >>> ZZZ|ABCDE
            ```

        Args:
            clip: Input clip.
            left: Padding added to the left side of the clip.
            right: Padding added to the right side of the clip.
            top: Padding added to the top side of the clip.
            bottom: Padding added to the bottom side of the clip.
            color: Constant color that should be added on the sides:

                   * number: This will be treated as such and not converted or clamped.
                   * False: Lowest value for this clip format and color range.
                   * True: Highest value for this clip format and color range.
                   * None: Neutral value for this clip format.
                   * MISSING: Automatically set to False if RGB, else None.

        Returns:
            Padded clip with colored borders.
        """
        return Padder(left, right, top, bottom).color(clip, color)

    mod_padding = Padder.mod_padding
    mod_padding_crop = Padder.mod_padding_crop


def normalize_planes(clip: vs.VideoNode, planes: Planes = None) -> list[int]:
    """
    Normalize a sequence of planes.

    Args:
        clip: Input clip.
        planes: Array of planes. If None, returns all planes of the input clip's format. Default: None.

    Returns:
        Sorted list of planes.
    """

    planes = list(range(clip.format.num_planes)) if planes is None or planes == 4 else to_arr(planes)

    return sorted(set(planes).intersection(range(clip.format.num_planes)))


def invert_planes(clip: vs.VideoNode, planes: Planes = None) -> list[int]:
    """
    Invert a sequence of planes.

    Args:
        clip: Input clip.
        planes: Array of planes. If None, selects all planes of the input clip's format.

    Returns:
        Sorted inverted list of planes.
    """
    return sorted(set(normalize_planes(clip, None)) - set(normalize_planes(clip, planes)))


def normalize_param_planes[T](clip: vs.VideoNode, param: T | Sequence[T], planes: Planes, null: T) -> list[T]:
    """
    Normalize a value or sequence to a list mapped to the clip's planes.

    For any plane not included in `planes`, the corresponding output value is set to `null`.

    Args:
        clip: The input clip whose format and number of planes will be used to determine mapping.
        param: A single value or a sequence of values to normalize across the clip's planes.
        planes: The planes to apply the values to. Other planes will receive `null`.
        null: The default value to use for planes that are not included in `planes`.

    Returns:
        A list of length equal to the number of planes in the clip, with `param` values or `null`.
    """
    planes = normalize_planes(clip, planes)

    return [p if i in planes else null for i, p in enumerate(normalize_seq(param, clip.format.num_planes))]


@overload
def flatten[T](items: Iterable[Iterable[T]]) -> Iterator[T]: ...


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
