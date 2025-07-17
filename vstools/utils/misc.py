from __future__ import annotations

from contextlib import AbstractContextManager
from fractions import Fraction
from math import floor
from types import TracebackType
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence, cast, overload

import vapoursynth as vs
from jetpytools import MISSING, MissingT, P
from typing_extensions import Self

from ..enums import Align, BaseAlign
from ..exceptions import InvalidSubsamplingError
from ..functions import Keyframes, check_variable_format, clip_data_gather
from ..types import ConstantFormatVideoNode, VideoNodeT
from ..utils.cache import SceneBasedDynamicCache
from .info import get_video_format
from .props import get_props

__all__ = ["SceneAverageStats", "change_fps", "match_clip", "padder", "padder_ctx", "pick_func_stype", "set_output"]


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

    return new_fps_clip.std.FrameEval(lambda n: clip[round(n / factor)])


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
    from ..enums import Matrix
    from ..functions import check_variable

    assert check_variable(clip, match_clip)
    assert check_variable(ref, match_clip)

    if length:
        if clip.num_frames < ref.num_frames:
            clip = vs.core.std.Splice([clip, clip[-1] * (ref.num_frames - clip.num_frames)])
        else:
            clip = clip[: ref.num_frames]

    clip = clip.resize.Bicubic(ref.width, ref.height) if dimensions else clip

    if vformat:
        clip = clip.resize.Bicubic(format=ref.format.id, matrix=Matrix.from_video(ref))

    if matrices:
        clip = clip.std.SetFrameProps(
            **get_props(ref, ["_Matrix", "_Transfer", "_Primaries"], int, default=2, func=match_clip)
        )

    return clip.std.AssumeFPS(fpsnum=ref.fps.numerator, fpsden=ref.fps.denominator)


class padder_ctx(AbstractContextManager["padder_ctx"]):  # noqa: N801
    """
    Context manager for the padder class.
    """

    def __init__(self, mod: int = 8, min: int = 0, align: Align = Align.MIDDLE_CENTER) -> None:
        """
        Args:
            mod: The modulus used for calculations or constraints. Defaults to 8.
            min: The minimum value allowed or used as a base threshold. Defaults to 0.
            align: The alignment configuration. Defaults to Align.MIDDLE_CENTER.
        """
        self.mod = mod
        self.min = min
        self.align = align
        self.pad_ops = list[tuple[tuple[int, int, int, int], tuple[int, int]]]()

    def CROP(  # noqa: N802
        self, clip: vs.VideoNode, crop_scale: float | tuple[float, float] | None = None
    ) -> ConstantFormatVideoNode:
        """
        Crop a clip with the padding values.

        Args:
            clip: Input clip.
            crop_scale: Optional scale factor for the padding values. If None, no scaling is applied.

        Returns:
            Cropped clip.
        """
        (padding, sizes) = self.pad_ops.pop(0)

        if crop_scale is None:
            crop_scale = (clip.width / sizes[0], clip.height / sizes[1])

        crop_pad = padder._crop_padding(padder._get_sizes_crop_scale(clip, crop_scale)[1])

        return clip.std.Crop(*(x * y for x, y in zip(padding, crop_pad)))

    def MIRROR(self, clip: VideoNodeT) -> VideoNodeT:  # noqa: N802
        """
        Pad a clip with reflect mode. This will reflect the clip on each side.

        Args:
            clip: Input clip.

        Returns:
            Padded clip with reflected borders.
        """
        padding = padder.mod_padding(clip, self.mod, self.min, self.align)
        out = padder.MIRROR(clip, *padding)
        self.pad_ops.append((padding, (out.width, out.height)))
        return out

    def REPEAT(self, clip: vs.VideoNode) -> ConstantFormatVideoNode:  # noqa: N802
        """
        Pad a clip with repeat mode. This will simply repeat the last row/column till the end.

        Args:
            clip: Input clip.

        Returns:
            Padded clip with repeated borders.
        """
        padding = padder.mod_padding(clip, self.mod, self.min, self.align)
        out = padder.REPEAT(clip, *padding)
        self.pad_ops.append((padding, (out.width, out.height)))
        return out

    def COLOR(  # noqa: N802
        self, clip: vs.VideoNode, color: int | float | bool | None | Sequence[int | float | bool | None] = (False, None)
    ) -> ConstantFormatVideoNode:
        """
        Pad a clip with a constant color.

        Args:
            clip: Input clip.
            color: Constant color that should be added on the sides: * number: This will be treated as such and not
                converted or clamped. * False: Lowest value for this clip format and color range. * True: Highest value
                for this clip format and color range. * None: Neutral value for this clip format. * MISSING:
                Automatically set to False if RGB, else None.

        Returns:
            Padded clip with colored borders.
        """
        padding = padder.mod_padding(clip, self.mod, self.min, self.align)
        out = padder.COLOR(clip, *padding, color)
        self.pad_ops.append((padding, (out.width, out.height)))
        return out

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        return None


class padder:  # noqa: N801
    """
    Pad out the pixels on the sides by the given amount of pixels.
    """

    ctx = padder_ctx

    @staticmethod
    def _base(
        clip: vs.VideoNode, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0
    ) -> tuple[int, int, vs.VideoFormat, int, int]:
        from ..functions import check_variable

        assert check_variable(clip, "padder")

        width = clip.width + left + right
        height = clip.height + top + bottom

        fmt = get_video_format(clip)

        w_sub, h_sub = 1 << fmt.subsampling_w, 1 << fmt.subsampling_h

        if width % w_sub and height % h_sub:
            raise InvalidSubsamplingError(
                "padder", fmt, "Values must result in a mod congruent to the clip's subsampling ({subsampling})!"
            )

        return width, height, fmt, w_sub, h_sub

    @classmethod
    def MIRROR(cls, clip: VideoNodeT, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0) -> VideoNodeT:  # noqa: N802
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

        from ..utils import core

        width, height, *_ = cls._base(clip, left, right, top, bottom)

        padded = core.resize.Point(
            core.std.CopyFrameProps(clip, clip.std.BlankClip()),
            width,
            height,
            src_top=-top,
            src_left=-left,
            src_width=width,
            src_height=height,
        )
        return core.std.CopyFrameProps(padded, clip)

    @classmethod
    def REPEAT(  # noqa: N802
        cls, clip: vs.VideoNode, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0
    ) -> ConstantFormatVideoNode:
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

        *_, fmt, w_sub, h_sub = cls._base(clip, left, right, top, bottom)

        padded = clip.std.AddBorders(left, right, top, bottom)

        right, bottom = clip.width + left, clip.height + top

        pads = [
            (left, right, top, bottom),
            (left // w_sub, right // w_sub, top // h_sub, bottom // h_sub),
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

    @classmethod
    def COLOR(  # noqa: N802
        cls,
        clip: vs.VideoNode,
        left: int = 0,
        right: int = 0,
        top: int = 0,
        bottom: int = 0,
        color: int | float | bool | None | MissingT | Sequence[int | float | bool | None | MissingT] = (False, MISSING),
    ) -> ConstantFormatVideoNode:
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

        from ..functions import normalize_seq
        from ..utils import core, get_lowest_values, get_neutral_values, get_peak_values

        assert check_variable_format(clip, "padder")

        cls._base(clip, left, right, top, bottom)

        def _norm(colr: int | float | bool | None | MissingT) -> Sequence[int | float]:
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

        return core.std.AddBorders(clip, left, right, top, bottom, norm_colors)

    @staticmethod
    def _get_sizes_crop_scale(
        sizes: tuple[int, int] | vs.VideoNode, crop_scale: float | tuple[float, float]
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        if isinstance(sizes, vs.VideoNode):
            sizes = (sizes.width, sizes.height)

        if not isinstance(crop_scale, tuple):
            crop_scale = (crop_scale, crop_scale)

        return sizes, crop_scale  # type: ignore[return-value]

    @staticmethod
    def _crop_padding(crop_scale: tuple[int, int]) -> tuple[int, int, int, int]:
        return tuple(crop_scale[0 if i < 2 else 1] for i in range(4))  # type: ignore

    @classmethod
    def mod_padding(
        cls, sizes: tuple[int, int] | vs.VideoNode, mod: int = 16, min: int = 4, align: Align = Align.MIDDLE_CENTER
    ) -> tuple[int, int, int, int]:
        sizes, _ = cls._get_sizes_crop_scale(sizes, 1)
        ph, pv = (mod - (((x + min * 2) - 1) % mod + 1) for x in sizes)
        left, top = floor(ph / 2), floor(pv / 2)
        left, right, top, bottom = tuple(x + min for x in (left, ph - left, top, pv - top))

        if align & BaseAlign.TOP:
            bottom += top
            top = 0
        elif align & BaseAlign.BOTTOM:
            top += bottom
            bottom = 0

        if align & BaseAlign.LEFT:
            right += left
            left = 0
        elif align & BaseAlign.RIGHT:
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
        sizes, crop_scale = cls._get_sizes_crop_scale(sizes, crop_scale)
        padding = cls.mod_padding(sizes, mod, min, align)
        return padding, tuple(x * crop_scale[0 if i < 2 else 1] for x, i in enumerate(padding))  # type: ignore


def pick_func_stype(
    clip: vs.VideoNode, func_int: Callable[P, VideoNodeT], func_float: Callable[P, VideoNodeT]
) -> Callable[P, VideoNodeT]:
    """
    Pick the function matching the sample type of the clip's format.

    Args:
        clip: Input clip.
        func_int: Function to run on integer clips.
        func_float: Function to run on float clips.

    Returns:
        Function matching the sample type of your clip's format.
    """

    assert check_variable_format(clip, pick_func_stype)

    return func_float if clip.format.sample_type == vs.FLOAT else func_int


@overload
def set_output(node: vs.VideoNode, index: int = ..., /, *, alpha: vs.VideoNode | None = ..., **kwargs: Any) -> None:
    """
    Set output node with optional index, and if available, use vspreview set_output.

    Args:
        node: Output node
        index: Index number, defaults to current maximum index number + 1 or 0 if no ouput exists yet
        alpha: Alpha planes node, defaults to None
        **kwargs: Additional arguments to be passed to `vspreview.set_output`.
    """


@overload
def set_output(
    node: vs.VideoNode, name: str | bool | None = ..., /, *, alpha: vs.VideoNode | None = ..., **kwargs: Any
) -> None:
    """
    Set output node with optional name, and if available, use vspreview set_output.

    Args:
        node: Output node
        name: Node's name, defaults to variable name
        alpha: Alpha planes node, defaults to None
        **kwargs: Additional arguments to be passed to `vspreview.set_output`.
    """


@overload
def set_output(
    node: vs.VideoNode,
    index: int = ...,
    name: str | bool | None = ...,
    /,
    alpha: vs.VideoNode | None = ...,
    **kwargs: Any,
) -> None:
    """
    Set output node with optional index and name, and if available, use vspreview set_output.

    Args:
        node: Output node.
        index: Index number, defaults to current maximum index number + 1 or 0 if no ouput exists yet.
        name: Node's name, defaults to variable name
        alpha: Alpha planes node, defaults to None.
        **kwargs: Additional arguments to be passed to `vspreview.set_output`.
    """


@overload
def set_output(node: vs.AudioNode, index: int = ..., /, **kwargs: Any) -> None:
    """
    Set output node with optional index, and if available, use vspreview set_output.

    Args:
        node: Output node.
        index: Index number, defaults to current maximum index number + 1 or 0 if no ouput exists yet.
        **kwargs: Additional arguments to be passed to `vspreview.set_output`.
    """


@overload
def set_output(node: vs.AudioNode, name: str | bool | None = ..., /, **kwargs: Any) -> None:
    """
    Set output node with optional name, and if available, use vspreview set_output.

    Args:
        node: Output node.
        name: Node's name, defaults to variable name
        **kwargs: Additional arguments to be passed to `vspreview.set_output`.
    """


@overload
def set_output(node: vs.AudioNode, index: int = ..., name: str | bool | None = ..., /, **kwargs: Any) -> None:
    """
    Set output node with optional index and name, and if available, use vspreview set_output.

    Args:
        node: Output node.
        index: Index number, defaults to current maximum index number + 1 or 0 if no ouput exists yet.
        name: Node's name, defaults to variable name.
        **kwargs: Additional arguments to be passed to `vspreview.set_output`.
    """


@overload
def set_output(
    node: Iterable[vs.RawNode | Iterable[vs.RawNode | Iterable[vs.RawNode]]],
    index: int | Sequence[int] = ...,
    /,
    **kwargs: Any,
) -> None:
    """
    Set output node with optional index, and if available, use vspreview set_output.

    Args:
        node: Output node.
        index: Index number, defaults to current maximum index number + 1 or 0 if no ouput exists yet.
        **kwargs: Additional arguments to be passed to `vspreview.set_output`.
    """


@overload
def set_output(
    node: Iterable[vs.RawNode | Iterable[vs.RawNode | Iterable[vs.RawNode]]],
    name: str | bool | None = ...,
    /,
    **kwargs: Any,
) -> None:
    """
    Set output node with optional name, and if available, use vspreview set_output.

    Args:
        node: Output node.
        name: Node's name, defaults to variable name.
        **kwargs: Additional arguments to be passed to `vspreview.set_output`.
    """


@overload
def set_output(
    node: Iterable[vs.RawNode | Iterable[vs.RawNode | Iterable[vs.RawNode]]],
    index: int | Sequence[int] = ...,
    name: str | bool | None = ...,
    /,
    **kwargs: Any,
) -> None:
    """
    Set output node with optional index and name, and if available, use vspreview set_output.

    Args:
        node: Output node.
        index: Index number, defaults to current maximum index number + 1 or 0 if no ouput exists yet
        name: Node's name, defaults to variable name
        **kwargs: Additional arguments to be passed to `vspreview.set_output`.
    """


def set_output(
    node: vs.RawNode | Iterable[vs.RawNode | Iterable[vs.RawNode | Iterable[vs.RawNode]]],
    index_or_name: int | Sequence[int] | str | bool | None = None,
    name: str | bool | None = None,
    /,
    alpha: vs.VideoNode | None = None,
    **kwargs: Any,
) -> None:
    """
    Set output node with optional index and name, and if available, use vspreview set_output.

    Args:
        node: Output node.
        index_or_name: Index number, defaults to current maximum index number + 1 or 0 if no ouput exists yet.
        name: Node's name, defaults to variable name
        alpha: Alpha planes node, defaults to None.
        **kwargs: Additional arguments to be passed to `vspreview.set_output`.
    """
    from ..functions import flatten, to_arr

    if isinstance(index_or_name, (str, bool)):
        index = None
        if not TYPE_CHECKING and isinstance(name, vs.VideoNode):
            # Backward compatible with older api
            alpha = name
        name = index_or_name
    else:
        index = index_or_name

    ouputs = vs.get_outputs()
    nodes = list[vs.RawNode](flatten(node)) if isinstance(node, Iterable) else [node]

    index = to_arr(index) if index is not None else [max(ouputs, default=-1) + 1]

    while len(index) < len(nodes):
        index.append(index[-1] + 1)

    try:
        from vspreview import set_output as vsp_set_output

        vsp_set_output(nodes, index, name, alpha=alpha, f_back=2, force_preview=True, **kwargs)
    except ModuleNotFoundError:
        for idx, n in zip(index, nodes):
            n.set_output(idx)


class SceneAverageStats(SceneBasedDynamicCache):
    _props_keys = ("Min", "Max", "Average")

    class cache(dict[int, tuple[float, float, float]]):  # noqa: N801
        def __init__(self, clip: vs.VideoNode, keyframes: Keyframes, plane: int) -> None:
            self.props = clip.std.PlaneStats(plane=plane)
            self.keyframes = keyframes

        def __getitem__(self, idx: int) -> tuple[float, float, float]:
            if idx not in self:
                frame_range = self.keyframes.scenes[idx]
                cut_clip = self.props[frame_range.start : frame_range.stop]

                frames_min_max_avg = clip_data_gather(
                    cut_clip,
                    None,
                    lambda n, f: tuple(cast(float, f.props[f"PlaneStats{p}"]) for p in SceneAverageStats._props_keys),
                )

                frames_min, frames_max, frames_avgs = [[x[i] for x in frames_min_max_avg] for i in (0, 1, 2)]

                self[idx] = (min(frames_min), max(frames_max), sum(frames_avgs) / len(frames_avgs))

            return super().__getitem__(idx)

    def __init__(
        self,
        clip: vs.VideoNode,
        keyframes: Keyframes | str,
        prop: str = "SceneStats",
        plane: int = 0,
        cache_size: int = 5,
    ) -> None:
        super().__init__(clip, keyframes, cache_size)

        self.prop_keys = tuple(f"{prop}{x}" for x in self._props_keys)
        self.scene_avgs = self.__class__.cache(self.clip, self.keyframes, plane)

    def get_clip(self, key: int) -> vs.VideoNode:
        return self.clip.std.SetFrameProps(**dict(zip(self.prop_keys, self.scene_avgs[key])))
