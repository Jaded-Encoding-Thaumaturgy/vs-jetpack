"""
This module provides a Pythonic interface for building and evaluating complex VapourSynth expressions
using standard Python syntax.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import cached_property
from typing import Any, Iterator, Sequence, SupportsIndex, overload

from jetpytools import CustomValueError, to_arr
from typing_extensions import Self

from vstools import HoldsVideoFormatT, VideoFormatT, get_video_format, vs, vs_object

from ..funcs import expr_func
from ..util import ExprVars
from .operators import ExprOperators
from .variables import ClipVar, ComputedVar, ExprVar

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
            # ie.clips[0] is clip_a, ie.clips[1] is clip_b
            average = (ie.clips[0] + ie.clips[1]) / 2
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
            sum_clips = ie.clips[0]

            for i in range(1, len(ie.clips)):
                sum_clips = sum_clips + ie.clips[i]

            average = sum_clips / len(ie.clips)
            ie.out = average  # -> "x y + z + a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p + q + 20 /"

        result = ie.clip
        ```

    - Example (complex): Unsharp mask implemented in [inline_expr][vsexprtools.inline_expr].
      Extended with configurable anti-ringing and anti-aliasing, and frequency-based limiting.
        ```py
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
                x = ie.clips[0]

                # Calculate blur for sharpening base
                blur = x[-1, -1] + x[-1, 0] + x[-1, 1] + x[0, -1] + x[0, 0] + x[0, 1] + x[1, -1] + x[1, 0] + x[1, 1]
                blur = blur / 9

                # Calculate sharpening amount
                sharp_diff = (x - blur) * strength
                effective_sharp_diff = sharp_diff

                # Apply low-frequency only processing if parameter > 0
                if low_freq.freq_limit > 0:
                    # Calculate high-frequency component by comparing local variance to a larger area
                    wider_blur = (
                        blur + x[-2, -2] + x[-2, 0] + x[-2, 2] + x[0, -2] + x[0, 2] + x[2, -2] + x[2, 0] + x[2, 2]
                    ) / 9
                    high_freq_indicator = ie.op.ABS(blur - wider_blur)

                    # Calculate texture complexity (higher in detailed areas,
                    # lower in flat areas)
                    texture_complexity = ie.op.MAX(ie.op.ABS(x - blur), ie.op.ABS(blur - wider_blur))

                    # Reduce sharpening in areas with high frequency content
                    # but low texture complexity
                    freq_ratio = ie.op.MAX(high_freq_indicator / (texture_complexity + 0.01), 0)
                    low_freq_factor = 1.0 - ie.op.MIN(
                        freq_ratio * low_freq.freq_ratio_scale * low_freq.freq_limit, low_freq.max_reduction
                    )

                    # Apply additional limiting for high-frequency content
                    # to effective_sharp_diff
                    effective_sharp_diff = effective_sharp_diff * low_freq_factor

                # Get horizontal neighbors from the original clip
                n1, n2, n3 = x[-1, -1], x[-1, 0], x[-1, 1]
                n4, n5 = x[0, -1], x[0, 1]
                n6, n7, n8 = x[1, -1], x[1, 0], x[1, 1]

                # Calculate minimum through pairwise comparisons
                min1 = ie.op.MIN(n1, n2)
                min2 = ie.op.MIN(min1, n3)
                min3 = ie.op.MIN(min2, n4)
                min4 = ie.op.MIN(min3, n5)
                min5 = ie.op.MIN(min4, n6)
                min6 = ie.op.MIN(min5, n7)
                local_min = ie.op.as_var(ie.op.MIN(min6, n8))

                # Calculate maximum through pairwise comparisons
                max1 = ie.op.MAX(n1, n2)
                max2 = ie.op.MAX(max1, n3)
                max3 = ie.op.MAX(max2, n4)
                max4 = ie.op.MAX(max3, n5)
                max5 = ie.op.MAX(max4, n6)
                max6 = ie.op.MAX(max5, n7)
                local_max = ie.op.as_var(ie.op.MAX(max6, n8))

                # Only calculate adaptive limiting if limit > 0
                if limit > 0:
                    # Calculate local variance to detect edges (high variance = potential aliasing)
                    variance = (
                        (n1 - x) ** 2
                        + (n2 - x) ** 2
                        + (n3 - x) ** 2
                        + (n4 - x) ** 2
                        + (n5 - x) ** 2
                        + (n6 - x) ** 2
                        + (n7 - x) ** 2
                        + (n8 - x) ** 2
                    )
                    variance = variance / 8

                    # Calculate edge detection using Sobel-like operators
                    h_edge = ie.op.ABS(n1 + 2 * n2 + n3 - n6 - 2 * n7 - n8)
                    v_edge = ie.op.ABS(n1 + 2 * n4 + n6 - n3 - 2 * n5 - n8)
                    edge_strength = ie.op.SQRT(h_edge**2 + v_edge**2)

                    # Adaptive sharpening strength based on edge detection and variance
                    # Reduce sharpening in high-variance areas to prevent aliasing
                    edge_factor = 1.0 - ie.op.MIN(edge_strength * 0.01, limit)
                    var_factor = 1.0 - ie.op.MIN(variance * 0.005, limit)
                    adaptive_strength = edge_factor * var_factor

                    # Apply adaptive sharpening to the effective_sharp_diff
                    effective_sharp_diff = effective_sharp_diff * adaptive_strength

                    # Clamp the sharp_diff to the local min and max to prevent ringing
                    final_output = ie.op.CLAMP(x + effective_sharp_diff, local_min, local_max)
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
        **kwargs: Additional keyword arguments passed to [expr_func][vsexprtools.expr_func].

    Yields:
        InlineExprWrapper object containing clip variables and operators.

            - The [clips][vsexprtools.inline.manager.InlineExprWrapper.clips] attribute is a sequence
              of [ClipVar][vsexprtools.inline.variables.ClipVar] objects, one for each input clip.
              These objects overload standard Python operators (`+`, `-`, `*`, `/`, `**`, `==`, `<`, `>` etc.)
              to build the expression. They also have helpful properties:

                 * `.peak`, `.neutral`, `.lowest`: Bitdepth-aware values.
                 * `.width`, `.height`, `.depth`: Clip properties.
                 * `[x, y]`: Relative pixel access (e.g., `clip[1, 0]` for the pixel to the right).
                 * `props`: Access to frame properties (e.g. `clip.props.PlaneStatsMax`).

            - The [op][vsexprtools.inline.manager.InlineExprWrapper.op] attribute is an object providing access
              to all `Expr` operators such as `op.CLAMP(value, min, max)`, `op.SQRT(value)`,
              `op.TERN(condition, if_true, if_false)`, etc.

            You must assign the final [ComputedVar][vsexprtools.inline.variables.ComputedVar]
            (the result of your expression) to `ie.out`.

            Additionnaly, you can use `print(ie.out)` to see the computed expression string.
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
            disable_poly()

    ie._compute_expr()


class InlineExprWrapper(tuple[Sequence[ClipVar], ExprOperators, "InlineExprWrapper"], vs_object):
    """
    A wrapper class for constructing and evaluating VapourSynth expressions inline using Python syntax.

    This class is intended to be used within the [inline_expr][vsexprtools.inline_expr] context manager
    and serves as the interface through which you build expressions using overloaded Python operators
    and expressive constructs.

    It provides access to input clips as [ClipVar][vsexprtools.inline.variables.ClipVar] instances,
    expression operators, and the final output clip.

    All expressions are constructed in a high-level, readable Python syntax that is internally translated to
    VapourSynth-compatible expression strings.

    Usage:
    ```py
    with inline_expr([clip_a, clip_b]) as ie:
        avg = (ie.clips[0] + ie.clips[1]) / 2
        ie.out = avg

    result = ie.clip
    ```

    Note:
        The `InlineExprWrapper` also behaves like a tuple containing:

        - The clip variables (`clips`).
        - Expression operator functions (`op`).
        - The wrapper itself (`self`).

        This allows unpacking like:
            `clips, op, self = ie`
    """

    op = ExprOperators()
    """
    [ExprOperators][vsexprtools.inline.operators.ExprOperators] object providing access to all `Expr` operators.
    """

    def __new__(cls, clips: Sequence[vs.VideoNode], format: HoldsVideoFormatT | VideoFormatT | None = None) -> Self:
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
        self._final_expr_node = self.clips[0].as_var()
        self._inner = (self.clips, self.op, self)
        self._iter = iter(self._inner)
        self._final_clip: vs.VideoNode | None = None

    @overload
    def __getitem__(self, i: SupportsIndex, /) -> Sequence[ClipVar] | ExprOperators | InlineExprWrapper: ...
    @overload
    def __getitem__(
        self, i: slice[Any, Any, Any], /
    ) -> tuple[Sequence[ClipVar] | ExprOperators | InlineExprWrapper]: ...

    def __getitem__(self, i: SupportsIndex | slice[Any, Any, Any]) -> Any:
        return self._inner[i]

    def __iter__(self) -> Iterator[Sequence[ClipVar] | ExprOperators | InlineExprWrapper]:
        return self._iter

    def __next__(self) -> Any:
        return next(self._iter)

    def _compute_expr(self, **kwargs: Any) -> None:
        self._final_clip = expr_func(
            self._nodes, [self._final_expr_node.to_str(plane=p) for p in range(self._format.num_planes)], **kwargs
        )

    @cached_property
    def clips(self) -> Sequence[ClipVar]:
        """
        Sequence of [ClipVar][vsexprtools.inline.variables.ClipVar] objects, one for each input clip.

        These objects overload standard Python operators (`+`, `-`, `*`, `/`, `**`, `==`, `<`, `>` etc.)
        to build the expression. They also have helpful properties:

        - `.peak`, `.neutral`, `.lowest`: Bitdepth-aware values.
        - `.width`, `.height`, `.depth`: Clip properties.
        - `[x, y]`: Relative pixel access (e.g., `clip[1, 0]` for the pixel to the right).
        - `props`: Access to frame properties (e.g. `clip.props.PlaneStatsMax`).

        See [ClipVar][vsexprtools.inline.variables.ClipVar] for all the possible properties.

        Returns:
            Sequence of [ClipVar][vsexprtools.inline.variables.ClipVar] objects.
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
    def out(self, out_var: ExprVar) -> None:
        """
        Set the final output of the expression.

        Converts the given [ExprVar][vsexprtools.inline.variables.ExprVar]
        to a [ComputedVar][vsexprtools.inline.variables.ComputedVar] and stores it as the final expression.
        """
        self._final_expr_node = ExprOperators.as_var(out_var)

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
