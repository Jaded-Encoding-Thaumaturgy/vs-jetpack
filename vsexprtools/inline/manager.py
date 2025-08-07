"""
This module provides a Pythonic interface for building and evaluating complex VapourSynth expressions
using standard Python syntax.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import cache
from typing import Any, Iterable, Iterator, Sequence, SupportsIndex, cast, overload

from jetpytools import CustomValueError, to_arr
from typing_extensions import Self

from vstools import HoldsVideoFormatT, VideoFormatT, get_video_format, vs, vs_object

from ..funcs import norm_expr
from ..util import ExprVars
from .helpers import ClipVar, ComputedVar, ExprVarLike, Operators, Tokens

__all__ = ["inline_expr"]


@contextmanager
def inline_expr(
    clips: vs.VideoNode | Sequence[vs.VideoNode],
    format: HoldsVideoFormatT | VideoFormatT | None = None,
    *,
    enable_polyfills: bool = False,
    **kwargs: Any,
) -> Iterator[InlineExprWrapper]:
    """
    A context manager for building and evaluating VapourSynth expressions in a Pythonic way.

    This function allows you to write complex VapourSynth expressions using standard Python
    operators and syntax, abstracting away the underlying RPN (Reverse Polish Notation) string.

    - <https://www.vapoursynth.com/doc/functions/video/expr.html>
    - <https://github.com/AkarinVS/vapoursynth-plugin/wiki/Expr>

    The context manager is initialized with one or more VapourSynth clips and yields a
    [InlineExprWrapper][vsexprtools.inline.manager.InlineExprWrapper] object containing clip variables and operators.

    Usage:
    ```py
    with inline_expr(clips) as ie:
        # ... build your expression here ...
        ie.out = ...

    # The final, processed clip is available after the context block.
    result_clip = ie.clip
    ```

    - Example (simple): Averaging two clips
        ```py
        from vsexprtools import inline_expr
        from vstools import core, vs

        clip_a = core.std.BlankClip(format=vs.YUV420P8, color=[255, 0, 0])
        clip_b = core.std.BlankClip(format=vs.YUV420P8, color=[0, 255, 0])

        with inline_expr([clip_a, clip_b]) as ie:
            # ie.vars[0] is clip_a, ie.vars[1] is clip_b
            average = (ie.vars[0] + ie.vars[1]) / 2
            ie.out = average

        result = ie.clip
        ```

    - Example (simple): Averaging 20 random clips
        ```py
        from vsexprtools import inline_expr
        from vstools import core, vs
        import random


        def spawn_random(amount: int) -> list[vs.VideoNode]:
            clips = list[vs.VideoNode]()

            for _ in range(amount):
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                clips.append(core.std.BlankClip(format=vs.RGB24, color=[r, g, b]))

            return clips


        with inline_expr(spawn_random(20)) as ie:
            ie.out = sum(ie.vars) / len(ie.vars)
            # -> "x y + z + a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p + q + 20 /"

        result = ie.clip
        ```

    - Example (advanced): [prefilter_to_full_range][vsdenoise.prefilter_to_full_range] implemented with `inline_expr`
        ```py
        from vsexprtools import inline_expr
        from vstools import ColorRange, vs


        def pf_full(clip: vs.VideoNode, slope: float = 2.0, smooth: float = 0.0625) -> vs.VideoNode:
            with inline_expr(clip) as ie:
                x, *_ = ie.vars

                # Normalize luma to 0-1 range
                norm_luma = (x - x.LumaRangeInMin) * (ie.as_var(1) / (x.LumaRangeInMax - x.LumaRangeInMin))
                # Ensure normalized luma stays within bounds
                norm_luma = ie.op.clamp(norm_luma, 0, 1)

                # Curve factor controls non-linearity based on slope and smoothing
                curve_strength = (slope - 1) * smooth  # Slope increases contrast in darker regions

                # Compute a non-linear boost that emphasizes dark details without crushing blacks
                nonlinear_boost = curve_strength * ((1 + smooth) - (ie.op.sin(smooth) / (norm_luma + smooth)))

                # Combine the non-linear boost with the normalized luma
                # Boosts shadows while preserving highlights
                weight_mul = nonlinear_boost + norm_luma * (1 - curve_strength)

                # Scale the final result back to the output range
                weight_mul *= x.RangeMax

                # Assign the processed luma to the Y plane
                ie.out.y = weight_mul

                # Round only if the format is integer
                if clip.format.sample_type is vs.INTEGER:
                    ie.out.y = ie.op.round(ie.out.y)

                if ColorRange.from_video(clip).is_full or clip.format.sample_type is vs.FLOAT:
                    ie.out.uv = x
                else:
                    # Scale chroma values from limited to full range:
                    #  - Subtract neutral chroma (e.g., 128) to center around zero
                    #  - Scale based on the input chroma range (e.g., 240 - 16)
                    #  - Add half of full range to re-center in full output range
                    chroma_mult = x.RangeMax / (x.ChromaRangeInMax - x.ChromaRangeInMin)
                    chroma_boosted = (x - x.Neutral) * chroma_mult + x.RangeHalf

                    # Apply the adjusted chroma values to U and V planes
                    ie.out.uv = ie.op.round(chroma_boosted)

            # Final output is flagged as full-range video
            return ColorRange.FULL.apply(ie.clip)
        ```

    - Example (complex): Unsharp mask implemented in [inline_expr][vsexprtools.inline_expr].
      Extended with configurable anti-ringing and anti-aliasing, and frequency-based limiting.
        ```py
        import functools
        import itertools
        from dataclasses import dataclass

        from vsexprtools import inline_expr
        from vsmasktools import Sobel
        from vstools import ConvMode, vs


        @dataclass
        class LowFreqSettings:
            freq_limit: float = 0.1
            freq_ratio_scale: float = 5.0
            max_reduction: float = 0.95


        def unsharp_limited(
            clip: vs.VideoNode,
            strength: float = 1.5,
            limit: float = 0.3,
            low_freq: LowFreqSettings = LowFreqSettings(freq_limit=0.1),
        ) -> vs.VideoNode:
            with inline_expr(clip) as ie:
                x = ie.vars[0]

                # Calculate blur for sharpening base
                blur = ie.op.convolution(x, [1] * 9)

                # Calculate sharpening amount
                sharp_diff = (x - blur) * strength
                effective_sharp_diff = sharp_diff

                # Apply low-frequency only processing if parameter > 0
                if low_freq.freq_limit > 0:
                    # Calculate high-frequency component by comparing local variance to a larger area
                    wider_blur = sum(x[i, j] for i, j in itertools.product([-2, 0, 2], repeat=2) if (i, j) != (0, 0))
                    wider_blur = ie.as_var(wider_blur) / 9
                    high_freq_indicator = abs(blur - wider_blur)

                    # Calculate texture complexity (higher in detailed areas,
                    # lower in flat areas)
                    texture_complexity = ie.op.max(abs(x - blur), abs(blur - wider_blur))

                    # Reduce sharpening in areas with high frequency content
                    # but low texture complexity
                    freq_ratio = ie.op.max(high_freq_indicator / (texture_complexity + 0.01), 0)
                    low_freq_factor = 1.0 - ie.op.min(
                        freq_ratio * low_freq.freq_ratio_scale * low_freq.freq_limit, low_freq.max_reduction
                    )

                    # Apply additional limiting for high-frequency content
                    # to effective_sharp_diff
                    effective_sharp_diff = effective_sharp_diff * low_freq_factor

                # Get horizontal neighbors from the original clip
                neighbors = [ie.as_var(x) for x in ie.op.matrix(x, 1, ConvMode.SQUARE, [(0, 0)])]

                # Calculate minimum
                local_min = functools.reduce(ie.op.min, neighbors)

                # Calculate maximum
                local_max = functools.reduce(ie.op.max, neighbors)

                # Only calculate adaptive limiting if limit > 0
                if limit > 0:
                    # Calculate local variance to detect edges (high variance = potential aliasing)
                    variance = sum(((n - x) ** 2 for n in neighbors)) / 8

                    # Calculate edge detection using Sobel-like operators
                    h_conv, v_conv = Sobel.matrices
                    h_edge = ie.op.convolution(x, h_conv, divisor=False, saturate=False)
                    v_edge = ie.op.convolution(x, v_conv, divisor=False, saturate=False)
                    edge_strength = ie.op.sqrt(h_edge**2 + v_edge**2)

                    # Adaptive sharpening strength based on edge detection and variance
                    # Reduce sharpening in high-variance areas to prevent aliasing
                    edge_factor = 1.0 - ie.op.min(edge_strength * 0.01, limit)
                    var_factor = 1.0 - ie.op.min(variance * 0.005, limit)
                    adaptive_strength = edge_factor * var_factor

                    # Apply adaptive sharpening to the effective_sharp_diff
                    effective_sharp_diff = effective_sharp_diff * adaptive_strength

                    # Clamp the sharp_diff to the local min and max to prevent ringing
                    final_output = ie.op.clamp(x + effective_sharp_diff, local_min, local_max)
                else:
                    # If limit is 0 or less, use the effective_sharp_diff (which might be basic or low-freq adjusted)
                    final_output = x + effective_sharp_diff

                # Set the final output
                ie.out = final_output

            return ie.clip
        ```

    Args:
        clips: Input clip(s).
        format: format: Output format, defaults to the first clip format.
        enable_polyfills: Enable monkey-patching built-in methods. Maybe more than that, nobody knows.
        **kwargs: Additional keyword arguments passed to [norm_expr][vsexprtools.norm_expr].

    Yields:
        InlineExprWrapper object containing clip variables and operators.

            - The [vars][vsexprtools.inline.manager.InlineExprWrapper.vars] attribute is a sequence
              of [ClipVar][vsexprtools.inline.helpers.ClipVar] objects, one for each input clip.
              These objects overload standard Python operators (`+`, `-`, `*`, `/`, `**`, `==`, `<`, `>` etc.)
              to build the expression.
              They also provide relative pixel access through the `__getitem__` dunder method
              (e.g., `x[1, 0]` for the pixel to the right),
              arbitrary access to frame properties (e.g. `x.props.PlaneStatsMax` or `x.props["PlaneStatsMax"]`)
              and bit-depth aware constants access (e.g., `x.RangeMax` or `x.Neutral`).


            - The [op][vsexprtools.inline.manager.InlineExprWrapper.op] attribute is an object providing access
              to all `Expr` operators such as `op.clamp(value, min, max)`, `op.sqrt(value)`,
              `op.tern(condition, if_true, if_false)`, etc.

            You must assign the final [ComputedVar][vsexprtools.inline.helpers.ComputedVar]
            (the result of your expression) to `ie.out`.

            Additionnaly, you can use `print(ie.out)` to see the computed expression string
            or `print(ie.out.to_str_per_plane())` or `print(ie.out.to_str(plane=...))` to see the expression per plane.
    """
    clips = to_arr(clips)
    ie = InlineExprWrapper(clips, format)

    try:
        if enable_polyfills:
            from .polyfills import disable_poly, enable_poly

            enable_poly()

        yield ie
    finally:
        if enable_polyfills:
            disable_poly()  # pyright: ignore[reportPossiblyUnboundVariable]

    kwargs.setdefault("func", inline_expr)

    ie._compute_expr(**kwargs)


class InlineExprWrapper(tuple[Sequence[ClipVar], Operators, "InlineExprWrapper"], vs_object):
    """
    A wrapper class for constructing and evaluating VapourSynth expressions inline using Python syntax.

    This class is intended to be used within the [inline_expr][vsexprtools.inline_expr] context manager
    and serves as the interface through which you build expressions using overloaded Python operators
    and expressive constructs.

    It provides access to input clips as [ClipVar][vsexprtools.inline.helpers.ClipVar] instances,
    expression operators, and the final output clip.

    All expressions are constructed in a high-level, readable Python syntax that is internally translated to
    VapourSynth-compatible expression strings.

    Usage:
    ```py
    with inline_expr([clip_a, clip_b]) as ie:
        avg = (ie.vars[0] + ie.vars[1]) / 2
        ie.out = avg

    result = ie.clip
    ```

    Note:
        The `InlineExprWrapper` also behaves like a tuple containing:

        - The clip variables (`vars`).
        - Expression operator functions (`op`).
        - The wrapper itself (`ie`).

        This allows unpacking like:
        ```py
        with inline_expr([clip_a, clip_b]) as (vars, op, ie):
            ...
        ```
    """

    op = Operators()
    """
    [Operators][vsexprtools.inline.helpers.Operators] object providing access to all `Expr` operators.
    """

    tk = Tokens()
    """
    [Tokens][vsexprtools.inline.helpers.Tokens] object providing access to all `Expr` tokens.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        return super().__new__(cls)

    def __init__(self, clips: Sequence[vs.VideoNode], format: HoldsVideoFormatT | VideoFormatT | None = None) -> None:
        """
        Initializes a new [InlineExprWrapper][vsexprtools.inline.manager.InlineExprWrapper] instance.

        Args:
            clips: Input clip(s).
            format: Output format, defaults to the first clip format.
        """
        self._nodes = clips
        self._format = get_video_format(format if format is not None else clips[0])
        self._final_expr_node = self.as_var("")
        self._inner = (self.vars, self.op, cast(Self, self))
        self._final_clip: vs.VideoNode | None = None

    @overload
    def __getitem__(self, i: SupportsIndex, /) -> Sequence[ClipVar] | Operators | Self: ...
    @overload
    def __getitem__(self, i: slice[Any, Any, Any], /) -> tuple[Sequence[ClipVar] | Operators | Self]: ...
    def __getitem__(self, i: SupportsIndex | slice[Any, Any, Any]) -> Any:
        return self._inner[i]

    def __iter__(self) -> Iterator[Sequence[ClipVar] | Operators | Self]:
        yield from self._inner

    def __next__(self) -> Any:
        return next(self)

    def _compute_expr(self, **kwargs: Any) -> None:
        self._final_clip = norm_expr(
            self._nodes,
            tuple(self._final_expr_node.to_str_per_plane(self._format.num_planes)),
            format=self._format,
            **kwargs,
        )

    @staticmethod
    def as_var(x: ExprVarLike | Iterable[ExprVarLike] = "") -> ComputedVar:
        """
        Converts an expression variable to a ComputedVar.

        Args:
            x: A single ExprVarLike or an Iterable of ExprVarLike.

        Returns:
            A ComputedVar.
        """
        return ComputedVar(x)

    @property
    @cache
    def vars(self) -> Sequence[ClipVar]:
        """
        Sequence of [ClipVar][vsexprtools.inline.helpers.ClipVar] objects, one for each input clip.

        These objects overload standard Python operators (`+`, `-`, `*`, `/`, `**`, `==`, `<`, `>` etc.)
        to build the expression.
        They also provide relative pixel access through the `__getitem__` dunder method
        (e.g., `x[1, 0]` for the pixel to the right),
        arbitrary access to frame properties (e.g. `x.props.PlaneStatsMax` or `x.props["PlaneStatsMax"]`)
        and bit-depth aware constants access (e.g., `x.RangeMax` or `x.Neutral`).

        Returns:
            Sequence of [ClipVar][vsexprtools.inline.helpers.ClipVar] objects.
        """
        return tuple(ClipVar(char, clip) for char, clip in zip(ExprVars.cycle(), self._nodes))

    @property
    def out(self) -> ComputedVar:
        """
        The final expression node representing the result of the expression.

        This is the computed expression that will be translated into a VapourSynth
        expression string. It must be assigned inside the context using `ie.out = ...`.
        """
        return self._final_expr_node

    @out.setter
    def out(self, out_var: ExprVarLike) -> None:
        """
        Set the final output of the expression.

        Converts the given [ExprVar][vsexprtools.inline.helpers.ExprVar]
        to a [ComputedVar][vsexprtools.inline.helpers.ComputedVar] and stores it as the final expression.
        """
        self._final_expr_node = self.as_var(out_var)

    @property
    def clip(self) -> vs.VideoNode:
        """
        The output VapourSynth clip generated from the final expression.

        This is only accessible after the context block has exited.

        Raises:
            CustomValueError: If accessed inside the context manager.

        Returns:
            The resulting clip after evaluating the expression.
        """
        if self._final_clip:
            return self._final_clip

        raise CustomValueError("You can only get the output clip out of the context manager!", self.__class__)

    def __vs_del__(self, core_id: int) -> None:
        del self._final_clip
        del self._nodes
        del self._format
