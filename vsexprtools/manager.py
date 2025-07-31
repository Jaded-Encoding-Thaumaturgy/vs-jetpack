"""
This module provides a Pythonic interface for building and evaluating complex VapourSynth expressions
using standard Python syntax.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from types import TracebackType
from typing import NamedTuple, Sequence

from vstools import vs

from .funcs import expr_func
from .operators import ExprOperators
from .polyfills import disable_poly, enable_poly
from .util import ExprVars
from .variables import ClipVar, ComputedVar, ExprVar

__all__ = ["InlineExpr", "inline_expr"]


class InlineExpr(NamedTuple):
    clips: list[ClipVar]
    op: ExprOperators
    out: inline_expr


class inline_expr(AbstractContextManager[InlineExpr]):  # noqa: N801
    """
    A context manager for building and evaluating VapourSynth expressions in a Pythonic way.

    This class allows you to write complex VapourSynth expressions using standard Python
    operators and syntax, abstracting away the underlying RPN (Reverse Polish Notation) string.

    - <https://www.vapoursynth.com/doc/functions/video/expr.html>
    - <https://github.com/AkarinVS/vapoursynth-plugin/wiki/Expr>

    The context manager is initialized with one or more VapourSynth clips and yields a
    tuple containing clip variables, operators, and the context manager instance itself.

    Yields:
        InlineExpr (InlineExpr): A tuple `(clips, op, ie)`:

               - `clips` is a a list of [ClipVar][vsexprtools.ClipVar] objects, one for each input clip.
                These objects overload standard Python operators (`+`, `-`, `*`, `/`, `**`, `==`, `<`, `>` etc.)
                to build the expression. They also have helpful properties:
                   - `.peak`, `.neutral`, `.lowest`: Bitdepth-aware values.
                   - `.width`, `.height`, `.depth`: Clip properties.
                   - `[x, y]`: Relative pixel access (e.g., `clip[1, 0]` for the pixel to the right).
                   - `props`: Access to frame properties (e.g. `clip.props.PlaneStatsMax`).
               - `op` is an object providing access to all `Expr` operators
                such as `op.CLAMP(value, min, max)`, `op.SQRT(value)`, `op.TERN(condition, if_true, if_false)`, etc.
               - `ie` is the context manager instance itself. You must assign the final `ComputedVar`
                (the result of your expression) to `ie.out`.

                You can use `print(ie.out)` to see computed expression string.

    Usage:
    ```py
    with inline_expr(clips) as (clips, op, ie):
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

        with inline_expr([clip_a, clip_b]) as (clips, op, ie):
            # clips[0] is clip_a, clips[1] is clip_b
            average = (clips[0] + clips[1]) / 2
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


        with inline_expr(spawn_random(20)) as (clips, op, ie):
            sum_clips = clips[0]

            for i in range(1, len(clips)):
                sum_clips = sum_clips + clips[i]

            average = sum_clips / len(clips)
            ie.out = average  # -> "x y + z + a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p + q + 20 /"

        result = ie.clip
        ```

    - Example (complex): Unsharp mask implemented in inline_expr.
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
            with inline_expr(clip) as (clips, op, ie):
                x = clips[0]

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
                    high_freq_indicator = op.ABS(blur - wider_blur)

                    # Calculate texture complexity (higher in detailed areas,
                    # lower in flat areas)
                    texture_complexity = op.MAX(op.ABS(x - blur), op.ABS(blur - wider_blur))

                    # Reduce sharpening in areas with high frequency content
                    # but low texture complexity
                    freq_ratio = op.MAX(high_freq_indicator / (texture_complexity + 0.01), 0)
                    low_freq_factor = 1.0 - op.MIN(
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
                min1 = op.MIN(n1, n2)
                min2 = op.MIN(min1, n3)
                min3 = op.MIN(min2, n4)
                min4 = op.MIN(min3, n5)
                min5 = op.MIN(min4, n6)
                min6 = op.MIN(min5, n7)
                local_min = op.as_var(op.MIN(min6, n8))

                # Calculate maximum through pairwise comparisons
                max1 = op.MAX(n1, n2)
                max2 = op.MAX(max1, n3)
                max3 = op.MAX(max2, n4)
                max4 = op.MAX(max3, n5)
                max5 = op.MAX(max4, n6)
                max6 = op.MAX(max5, n7)
                local_max = op.as_var(op.MAX(max6, n8))

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
                    h_edge = op.ABS(n1 + 2 * n2 + n3 - n6 - 2 * n7 - n8)
                    v_edge = op.ABS(n1 + 2 * n4 + n6 - n3 - 2 * n5 - n8)
                    edge_strength = op.SQRT(h_edge**2 + v_edge**2)

                    # Adaptive sharpening strength based on edge detection and variance
                    # Reduce sharpening in high-variance areas to prevent aliasing
                    edge_factor = 1.0 - op.MIN(edge_strength * 0.01, limit)
                    var_factor = 1.0 - op.MIN(variance * 0.005, limit)
                    adaptive_strength = edge_factor * var_factor

                    # Apply adaptive sharpening to the effective_sharp_diff
                    effective_sharp_diff = effective_sharp_diff * adaptive_strength

                    # Clamp the sharp_diff to the local min and max to prevent ringing
                    final_output = op.CLAMP(x + effective_sharp_diff, local_min, local_max)
                else:
                    # If limit is 0 or less, use the effective_sharp_diff (which might be basic or low-freq adjusted)
                    final_output = x + effective_sharp_diff

                # Set the final output
                ie.out = final_output

            return ie.clip
        ```
    """

    _clips: list[vs.VideoNode]
    _in_context: bool
    _final_clip: vs.VideoNode | None
    _final_expr_node: ComputedVar

    def __init__(self, clips: vs.VideoNode | Sequence[vs.VideoNode]) -> None:
        """
        Initializes the class.

        Args:
            clips: A single clip or a sequence of clip objects to be used in the expression.
                These will be accessible within the context as `ClipVar` objects.
        """
        self._in_context = False

        self._clips = list(clips) if isinstance(clips, Sequence) else [clips]
        self._clips_char_map = [ClipVar(char, clip, self) for char, clip in zip(ExprVars.cycle, self._clips)]  # pyright: ignore[reportArgumentType]

        self._final_clip = None
        self._final_expr_node = self._clips_char_map[0].as_var()

    def __enter__(self) -> InlineExpr:
        self._in_context = True

        enable_poly()

        return InlineExpr(self._clips_char_map, ExprOperators(), self)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        self._final_clip = self._get_clip()

        disable_poly()

        self._in_context = False

        return None

    def _get_clip(self) -> vs.VideoNode:
        fmt = self._clips[0].format
        assert fmt

        return expr_func(self._clips, [self._final_expr_node.to_str(plane=plane) for plane in range(fmt.num_planes)])

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

        Converts the given `ExprVar` to a `ComputedVar` and stores it as the final expression.
        """
        self._final_expr_node = ExprOperators.as_var(out_var)

    @property
    def clip(self) -> vs.VideoNode:
        """
        The output VapourSynth clip generated from the final expression.

        Raises:
            ValueError: If accessed inside the context manager.
            ValueError: If `out` was not assigned before exiting the context.
            ValueError: If an error occurred during evaluation.

        Returns:
            The resulting clip after evaluating the expression.
        """
        if self._in_context:
            raise ValueError("You can only get the output clip out of the context manager!")

        if self._final_expr_node is None:
            raise ValueError("inline_expr: you need to call `out` with the output node!")

        if self._final_clip is None:
            raise ValueError("inline_expr: can't get output clip if the manager errored!")

        return self._final_clip
