from __future__ import annotations

from enum import EnumMeta
from functools import cache
from itertools import cycle
from math import inf, isqrt
from typing import Any, Collection, Iterable, Iterator, Literal, Sequence, SupportsIndex, cast, overload

from jetpytools import CustomRuntimeError, CustomStrEnum, SupportsString, SupportsSumNoDefaultT
from typing_extensions import Self

from vstools import (
    ColorRange,
    ConstantFormatVideoNode,
    ConvMode,
    CustomIndexError,
    CustomValueError,
    FuncExceptT,
    HoldsVideoFormatT,
    PlanesT,
    StrList,
    VideoFormatT,
    VideoNodeIterableT,
    VideoNodeT,
    flatten,
    flatten_vnodes,
    get_lowest_value,
    get_neutral_value,
    get_peak_value,
    vs,
)

from .util import ExprVars, _get_akarin_expr_version

__all__ = ["ExprList", "ExprOp", "ExprToken", "TupleExprList"]


class ExprToken(CustomStrEnum):
    """
    Enumeration for symbolic constants used in [norm_expr][vsexprtools.norm_expr].
    """

    LumaMin = "ymin"
    """The minimum luma value in limited range."""

    ChromaMin = "cmin"
    """The minimum chroma value in limited range."""

    LumaMax = "ymax"
    """The maximum luma value in limited range."""

    ChromaMax = "cmax"
    """The maximum chroma value in limited range."""

    Neutral = "neutral"
    """The neutral value (e.g. 128 for 8-bit limited, 0 for float)."""

    RangeHalf = "range_half"
    """Half of the full range (e.g. 128.0 for 8-bit full range)."""

    RangeSize = "range_size"
    """The size of the full range (e.g. 256 for 8-bit, 65536 for 16-bit)."""

    RangeMin = "range_min"
    """Minimum value in full range (chroma-aware)."""

    LumaRangeMin = "yrange_min"
    """Minimum luma value based on input clip's color range."""

    ChromaRangeMin = "crange_min"
    """Minimum chroma value based on input clip's color range."""

    RangeMax = "range_max"
    """Maximum value in full range (chroma-aware)."""

    LumaRangeMax = "yrange_max"
    """Maximum luma value based on input clip's color range."""

    ChromaRangeMax = "crange_max"
    """Maximum chroma value based on input clip's color range."""

    RangeInMin = "range_in_min"
    """Like `RangeMin`, but adapts to input `range_in` parameter."""

    LumaRangeInMin = "yrange_in_min"
    """Like `LumaRangeMin`, but adapts to input `range_in`."""

    ChromaRangeInMin = "crange_in_min"
    """Like `ChromaRangeMin`, but adapts to input `range_in`."""

    RangeInMax = "range_in_max"
    """Like `RangeMax`, but adapts to input `range_in`."""

    LumaRangeInMax = "yrange_in_max"
    """Like `LumaRangeMax`, but adapts to input `range_in`."""

    ChromaRangeInMax = "crange_in_max"
    """Like `ChromaRangeMax`, but adapts to input `range_in`."""

    @property
    def is_chroma(self) -> bool:
        """
        Indicates whether the token refers to a chroma-related value.

        Returns:
            True if the token refers to chroma (e.g. ChromaMin), False otherwise.
        """
        return "chroma" in self._name_.lower()

    def get_value(self, clip: vs.VideoNode, chroma: bool | None = None, range_in: ColorRange | None = None) -> float:
        """
        Resolves the numeric value represented by this token based on the input clip and range.

        Args:
            clip: A clip used to determine bit depth and format.
            chroma: Optional override for whether to treat the token as chroma-related.
            range_in: Optional override for the color range.

        Returns:
            The value corresponding to the symbolic token.
        """
        match self:
            case ExprToken.LumaMin:
                return get_lowest_value(clip, False, ColorRange.LIMITED)

            case ExprToken.ChromaMin:
                return get_lowest_value(clip, True, ColorRange.LIMITED)

            case ExprToken.LumaMax:
                return get_peak_value(clip, False, ColorRange.LIMITED)

            case ExprToken.ChromaMax:
                return get_peak_value(clip, True, ColorRange.LIMITED)

            case ExprToken.Neutral:
                return get_neutral_value(clip)

            case ExprToken.RangeHalf:
                val = get_peak_value(clip, range_in=ColorRange.FULL)
                return (val + 1) / 2 if val > 1.0 else val

            case ExprToken.RangeSize:
                val = get_peak_value(clip, range_in=ColorRange.FULL)
                return val + 1 if val > 1.0 else val

            case ExprToken.RangeMin:
                return get_lowest_value(clip, chroma if chroma is not None else False, ColorRange.FULL)

            case ExprToken.LumaRangeMin:
                return get_lowest_value(clip, False)

            case ExprToken.ChromaRangeMin:
                return get_lowest_value(clip, True)

            case ExprToken.RangeMax:
                return get_peak_value(clip, chroma if chroma is not None else False, ColorRange.FULL)

            case ExprToken.LumaRangeMax:
                return get_peak_value(clip, False)

            case ExprToken.ChromaRangeMax:
                return get_peak_value(clip, True)

            case ExprToken.RangeInMin:
                return get_lowest_value(clip, chroma if chroma is not None else False, range_in)

            case ExprToken.LumaRangeInMin:
                return get_lowest_value(clip, False, range_in)

            case ExprToken.ChromaRangeInMin:
                return get_lowest_value(clip, True, range_in)

            case ExprToken.RangeInMax:
                return get_peak_value(clip, chroma if chroma is not None else False, range_in)

            case ExprToken.LumaRangeInMax:
                return get_peak_value(clip, False, range_in)

            case ExprToken.ChromaRangeInMax:
                return get_peak_value(clip, True, range_in)

    def __getitem__(self, i: int) -> str:  # type: ignore[override]
        """
        Returns a version of the token specific to a clip index.

        This allows referencing the token in expressions targeting multiple clips
        (e.g., `ExprToken.LumaMax[2]` results in `'ymax_z'` suffix for clip index 2).

        Args:
            i: An integer index representing the input clip number.

        Returns:
            An string with an index-specific suffix for use in expressions.
        """
        if i > 25:
            raise CustomIndexError("Only an index up to 25 is supported", self, i)

        return f"{self._value_}_{ExprVars.get_var(i)}"


class ExprList(StrList):
    """
    A list-based representation of a RPN expression.
    """

    def __call__(
        self,
        *clips: VideoNodeIterableT[vs.VideoNode],
        planes: PlanesT = None,
        format: HoldsVideoFormatT | VideoFormatT | None = None,
        opt: bool = False,
        boundary: bool = True,
        func: FuncExceptT | None = None,
        split_planes: bool = False,
        **kwargs: Any,
    ) -> ConstantFormatVideoNode:
        """
        Apply the expression to one or more input clips.

        Args:
            clips: Input clip(s).
            planes: Plane to process, defaults to all.
            format: Output format, defaults to the first clip format.
            opt: Forces integer evaluation as much as possible.
            boundary: Specifies the default boundary condition for relative pixel accesses:

                   - True (default): Mirrored edges.
                   - False: Clamped edges.
            func: Function returned for custom error handling. This should only be set by VS package developers.
            split_planes: Splits the VideoNodes into their individual planes.
            kwargs: Additional keyword arguments passed to [norm_expr][vsexprtools.norm_expr].

        Returns:
            Evaluated clip.
        """
        from .funcs import norm_expr

        return norm_expr(clips, self, planes, format, opt, boundary, func, split_planes, **kwargs)


class TupleExprList(tuple[ExprList, ...]):
    """
    A tuple of multiple `ExprList` expressions, applied sequentially to the clip(s).
    """

    def __call__(
        self,
        *clips: VideoNodeIterableT[vs.VideoNode],
        planes: PlanesT = None,
        format: HoldsVideoFormatT | VideoFormatT | None = None,
        opt: bool = False,
        boundary: bool = True,
        func: FuncExceptT | None = None,
        split_planes: bool = False,
        **kwargs: Any,
    ) -> ConstantFormatVideoNode:
        """
        Apply a sequence of expressions to the input clip(s), one after another.

        Each `ExprList` in the tuple is applied to the result of the previous one.

        Args:
            clips: Input clip(s).
            planes: Plane to process, defaults to all.
            format: Output format, defaults to the first clip format.
            opt: Forces integer evaluation as much as possible.
            boundary: Specifies the default boundary condition for relative pixel accesses:

                   - True (default): Mirrored edges.
                   - False: Clamped edges.
            func: Function returned for custom error handling. This should only be set by VS package developers.
            split_planes: Splits the VideoNodes into their individual planes.
            kwargs: Extra keyword arguments passed to each `ExprList`.

        Returns:
            Evaluated clip.

        Raises:
            CustomRuntimeError: If the `TupleExprList` is empty.
        """
        if len(self) < 1:
            raise CustomRuntimeError("You need at least one ExprList.", func, self)

        clip = flatten_vnodes(*clips)

        for exprlist in self:
            clip = exprlist(
                clip,
                planes=planes,
                format=format,
                opt=opt,
                boundary=boundary,
                func=func,
                split_planes=split_planes,
                **kwargs,
            )

        return clip[0] if isinstance(clip, Sequence) else clip  # type: ignore[return-value]

    def __str__(self) -> str:
        return str(tuple(str(e) for e in self))


class ExprOpBase(CustomStrEnum):
    """
    Base class for expression operators used in RPN expressions.
    """

    n_op: int
    """The number of operands the operator requires."""

    def __new__(cls, value: str, n_op: int) -> Self:
        self = str.__new__(cls, value)
        self._value_ = value
        self.n_op = n_op

        return self

    __str__ = str.__str__

    @overload
    def __call__(
        self,
        *clips: VideoNodeIterableT[VideoNodeT],
        suffix: SupportsString | Iterable[SupportsString] | None = None,
        prefix: SupportsString | Iterable[SupportsString] | None = None,
        expr_suffix: SupportsString | Iterable[SupportsString] | None = None,
        expr_prefix: SupportsString | Iterable[SupportsString] | None = None,
        planes: PlanesT = None,
        **kwargs: Any,
    ) -> VideoNodeT:
        """
        Combines multiple video clips using the selected expression operator.

        Args:
            clips: Input clip(s).
            suffix: Optional suffix string(s) to append to each input variable in the expression.
            prefix: Optional prefix string(s) to prepend to each input variable in the expression.
            expr_suffix: Optional expression to append after the combined input expression.
            expr_prefix: Optional expression to prepend before the combined input expression.
            planes: Which planes to process. Defaults to all.
            **kwargs: Additional keyword arguments forwarded to [combine][vsexprtools.combine].

        Returns:
            A clip representing the combined result of applying the expression.
        """

    @overload
    def __call__(self, *pos_args: Any, **kwargs: Any) -> str:
        """
        Returns a formatted version of the ExprOp, using substitutions from pos_args and kwargs.

        The substitutions are identified by braces ('{' and '}').

        Args:
            *pos_args: Positional arguments.
            **kwargs: Keywords arguments.

        Returns:
            Formatted version of this ExprOp.
        """

    def __call__(self, *pos_args: Any, **kwargs: Any) -> vs.VideoNode | str:
        """
        Combines multiple video clips using the selected expression operator
        or returns a formatted version of the ExprOp, using substitutions from pos_args and kwargs.

        Args:
            *pos_args: Positional arguments.
            **kwargs: Keywords arguments.

        Returns:
            A clip representing the combined result of applying the expression or formatted version of this ExprOp.
        """
        args = list(flatten(pos_args))

        if args and isinstance(args[0], vs.VideoNode):
            return self.combine(*args, **kwargs)

        while True:
            try:
                return self.format(*args, **kwargs)
            except KeyError as key:
                if not args:
                    raise
                kwargs[key.args[0]] = args.pop(0)

    def __next__(self) -> Self:
        return self

    def __iter__(self) -> Iterator[Self]:
        return cycle([self])

    def __mul__(self, n: int) -> list[Self]:  # type: ignore[override]
        return [self] * n

    def combine(
        self,
        *clips: vs.VideoNode | Iterable[vs.VideoNode | Iterable[vs.VideoNode]],
        suffix: SupportsString | Iterable[SupportsString] | None = None,
        prefix: SupportsString | Iterable[SupportsString] | None = None,
        expr_suffix: SupportsString | Iterable[SupportsString] | None = None,
        expr_prefix: SupportsString | Iterable[SupportsString] | None = None,
        planes: PlanesT = None,
        **kwargs: Any,
    ) -> ConstantFormatVideoNode:
        """
        Combines multiple video clips using the selected expression operator.

        Args:
            clips: Input clip(s).
            suffix: Optional suffix string(s) to append to each input variable in the expression.
            prefix: Optional prefix string(s) to prepend to each input variable in the expression.
            expr_suffix: Optional expression to append after the combined input expression.
            expr_prefix: Optional expression to prepend before the combined input expression.
            planes: Which planes to process. Defaults to all.
            **kwargs: Additional keyword arguments forwarded to [combine][vsexprtools.combine].

        Returns:
            A clip representing the combined result of applying the expression.
        """
        from .funcs import combine

        return combine(clips, self, suffix, prefix, expr_suffix, expr_prefix, planes, **kwargs)


class ExprOpExtraMeta(EnumMeta):
    @property
    def _extra_op_names_(cls) -> tuple[str, ...]:
        return (
            "SGN",
            "NEG",
            "TAN",
            "ATAN",
            "ASIN",
            "ACOS",
            "CEIL",
            "MMG",
            "LERP",
            "POLYVAL",
        )


class ExprOp(ExprOpBase, metaclass=ExprOpExtraMeta):
    """
    Represents operators used in RPN expressions.

    Each class attribute corresponds to a specific expression operator
    with its associated symbol and arity (number of required operands).

    Note: format strings can include placeholders for dynamic substitution (e.g., `{N:d}`, `{name:s}`).
    """

    # 0 Argument (akarin)
    N = "N", 0
    """Current frame number."""

    X = "X", 0
    """Current pixel X-coordinate."""

    Y = "Y", 0
    """Current pixel Y-coordinate."""

    WIDTH = "width", 0
    """Frame width."""

    HEIGHT = "height", 0
    """Frame height."""

    PI = "pi", 0
    """Mathematical constant π (pi)."""

    # 1 Argument (std)
    EXP = "exp", 1
    """Exponential function (e^x)."""

    LOG = "log", 1
    """Natural logarithm."""

    SQRT = "sqrt", 1
    """Square root."""

    SIN = "sin", 1
    """Sine (radians)."""

    COS = "cos", 1
    """Cosine (radians)."""

    ABS = "abs", 1
    """Absolute value."""

    NOT = "not", 1
    """Logical NOT."""

    DUP = "dup", 1
    """Duplicate the top of the stack."""

    DUPN = "dup{N:d}", 1
    """Duplicate the top N items on the stack."""

    # 1 Argument (akarin)
    TRUNC = "trunc", 1
    """Truncate to integer (toward zero)."""

    ROUND = "round", 1
    """Round to nearest integer."""

    FLOOR = "floor", 1
    """Round down to nearest integer."""

    BITNOT = "bitnot", 1
    """Bitwise NOT."""

    DROP = "drop", 1
    """Remove top value from the stack."""

    DROPN = "drop{N:d}", 1
    """Remove top N values from the stack."""

    SORTN = "sort{N:d}", 1
    """Sort top N values on the stack."""

    VAR_STORE = "{name:s}!", 1
    """Store value in variable named `{name}`."""

    VAR_PUSH = "{name:s}@", 1
    """Push value of variable `{name}` to the stack."""

    # 2 Arguments (std)
    MAX = "max", 2
    """Maximum of two values."""

    MIN = "min", 2
    """Minimum of two values."""

    ADD = "+", 2
    """Addition."""

    SUB = "-", 2
    """Subtraction."""

    MUL = "*", 2
    """Multiplication."""

    DIV = "/", 2
    """Division."""

    POW = "pow", 2
    """Exponentiation (x^y)."""

    GT = ">", 2
    """Greater than (x > y)."""

    LT = "<", 2
    """Less than (x < y)."""

    EQ = "=", 2
    """Equality (x == y)."""

    GTE = ">=", 2
    """Greater than or equal."""

    LTE = "<=", 2
    """Less than or equal."""

    AND = "and", 2
    """Logical AND."""

    OR = "or", 2
    """Logical OR."""

    XOR = "xor", 2
    """Logical XOR."""

    SWAP = "swap", 2
    """Swap top two values on the stack."""

    SWAPN = "swap{N:d}", 2
    """Swap the top N values (custom depth)."""

    # 2 Argument (akarin)
    MOD = "%", 2
    """Modulo operation (remainder)."""

    BITAND = "bitand", 2
    """Bitwise AND."""

    BITOR = "bitor", 2
    """Bitwise OR."""

    BITXOR = "bitxor", 2
    """Bitwise XOR."""

    # 3 Arguments (std)
    TERN = "?", 3
    """Ternary operation: cond ? if_true : if_false."""

    # 3 Argument (akarin)
    CLAMP = "clamp", 3
    """Clamp a value between min and max."""

    # Special Operators
    REL_PIX = "{char:s}[{x:d},{y:d}]", 3
    """Get value of relative pixel at offset ({x},{y}) on clip `{char}`."""

    ABS_PIX = "{x:d} {y:d} {char:s}[]", 3
    """Get value of absolute pixel at coordinates ({x},{y}) on clip `{char}`."""

    # Not Implemented in akarin or std
    SGN = "sgn", 1
    """Sign function: -1, 0, or 1 depending on value."""

    NEG = "neg", 1
    """Negation (multiply by -1)."""

    TAN = "tan", 1
    """Tangent (radians)."""

    ATAN = "atan", 1
    """Arctangent."""

    ASIN = "asin", 1
    """Arcsine (inverse sine)."""

    ACOS = "acos", 1
    """Arccosine (inverse cosine)."""

    CEIL = "ceil", 1
    """Round up to nearest integer."""

    MMG = "mmg", 3
    """MaskedMerge implementation from std lib."""

    # Implemented in akarin v0.96g but closed source and only available on Windows.
    LERP = "lerp", 3
    """Linear interpolation of a value between two border values."""

    POLYVAL = "polyval{N:d}", cast(int, inf)
    """
    Evaluate a degree-N polynomial at the top value on the stack.

    Uses N coefficients below the top value (x), ordered from highest to lowest degree.
    """

    @cache
    def is_extra(self) -> bool:
        """
        Check if the operator is an 'extra' operator.

        Extra operators are not natively supported by VapourSynth's `std.Expr` or `akarin.Expr`
        and require conversion to a valid equivalent expression.

        Returns:
            True if the operator is considered extra and requires conversion.
        """
        return self.name in ExprOp._extra_op_names_

    def convert_extra(  # type: ignore[misc]
        self: Literal[
            ExprOp.SGN,
            ExprOp.NEG,
            ExprOp.TAN,
            ExprOp.ATAN,
            ExprOp.ASIN,
            ExprOp.ACOS,
            ExprOp.CEIL,
            ExprOp.MMG,
            ExprOp.LERP,
            ExprOp.POLYVAL,
        ],  # pyright: ignore[reportGeneralTypeIssues]
        degree: int | None = None,
    ) -> str:
        """
        Converts an 'extra' operator into a valid `akarin.Expr` expression string.

        Args:
            degree: If calling from POLYVAL, the degree of the polynomial.

        Returns:
            A string representation of the equivalent expression.

        Raises:
            ValueError: If the operator is not marked as extra.
            NotImplementedError: If the extra operator has no defined conversion.
        """
        if not self.is_extra():
            raise CustomValueError

        match self:
            case ExprOp.SGN:
                return "dup 0 > swap 0 < -"
            case ExprOp.NEG:
                return "-1 *"
            case ExprOp.TAN:
                return "dup sin swap cos /"
            case ExprOp.ATAN:
                return self.atan().to_str()
            case ExprOp.ASIN:
                return self.asin().to_str()
            case ExprOp.ACOS:
                return self.acos().to_str()
            case ExprOp.CEIL:
                return "-1 * floor -1 *"
            case ExprOp.MMG:
                return self.masked_merge().to_str()
            case ExprOp.LERP:
                if bytes(self, "utf-8") in _get_akarin_expr_version()["expr_features"]:
                    return str(self)
                return "dup 1 - swap2 * swap2 * - __LERP! range_max 1 <= __LERP@ __LERP@ round ?"
            case ExprOp.POLYVAL:
                assert degree is not None
                return self.polyval("", *[""] * (degree + 1)).to_str()
            case _:
                raise NotImplementedError

    @classmethod
    def clamp(
        cls, min: float | ExprToken = ExprToken.RangeMin, max: float | ExprToken = ExprToken.RangeMax, c: str = ""
    ) -> ExprList:
        """
        Create an expression to clamp a value between `min` and `max`.

        Args:
            min: The minimum value.
            max: The maximum value.
            c: Optional expression variable or prefix to clamp.

        Returns:
            An `ExprList` containing the clamping expression.
        """
        return ExprList([c, min, max, ExprOp.CLAMP])

    @classmethod
    def matrix(
        cls,
        var: SupportsString | Collection[SupportsString],
        radius: int,
        mode: ConvMode,
        exclude: Iterable[tuple[int, int]] | None = None,
    ) -> TupleExprList:
        """
        Generate a matrix expression layout for convolution-like operations.

        Args:
            var: The variable representing the central pixel
                or elements proportional to the radius if mode is `Literal[ConvMode.TEMPORAL]`.
            radius: The radius of the kernel in pixels (e.g., 1 for 3x3).
            mode: The convolution mode.
            exclude: Optional set of (x, y) coordinates to exclude from the matrix.

        Returns:
            A [TupleExprList][vsexprtools.TupleExprList] representing the matrix of expressions.

        Raises:
            CustomValueError: If the input variable is not sized correctly for temporal mode.
            NotImplementedError: If the convolution mode is unsupported.
        """
        exclude = list(exclude) if exclude else []

        match mode:
            case ConvMode.SQUARE:
                coordinates = [(x, y) for y in range(-radius, radius + 1) for x in range(-radius, radius + 1)]
            case ConvMode.VERTICAL:
                coordinates = [(0, xy) for xy in range(-radius, radius + 1)]
            case ConvMode.HORIZONTAL:
                coordinates = [(xy, 0) for xy in range(-radius, radius + 1)]
            case ConvMode.HV:
                return TupleExprList(
                    [
                        cls.matrix(var, radius, ConvMode.VERTICAL, exclude)[0],
                        cls.matrix(var, radius, ConvMode.HORIZONTAL, exclude)[0],
                    ]
                )
            case ConvMode.TEMPORAL:
                assert isinstance(var, Collection)

                if len(var) != radius * 2 + 1:
                    raise CustomValueError(
                        "`var` must have a number of elements proportional to the radius", cls.matrix, var
                    )

                return TupleExprList([ExprList(v for v in var)])
            case _:
                raise NotImplementedError

        assert isinstance(var, SupportsString)

        return TupleExprList(
            [
                ExprList(
                    [
                        var if x == y == 0 else ExprOp.REL_PIX(var, x, y)
                        for (x, y) in coordinates
                        if (x, y) not in exclude
                    ]
                )
            ]
        )

    @classmethod
    def convolution(
        cls,
        var: SupportsString | Collection[SupportsString],
        matrix: Iterable[SupportsSumNoDefaultT] | Iterable[Iterable[SupportsSumNoDefaultT]],
        bias: SupportsString | None = None,
        divisor: SupportsString | bool = True,
        saturate: bool = True,
        mode: ConvMode = ConvMode.SQUARE,
        premultiply: SupportsString | None = None,
        multiply: SupportsString | None = None,
        clamp: bool = False,
    ) -> TupleExprList:
        """
        Builds an expression that performs a weighted convolution-like operation.

        Args:
            var: The variable used as the central value
                or elements proportional to the radius if mode is `Literal[ConvMode.TEMPORAL]`.
            matrix: A flat or 2D iterable representing the convolution weights.
            bias: A constant value to add to the result after convolution (default: None).
            divisor: If True, normalizes by the sum of weights; if False, skips division;
                Otherwise, divides by this value.
            saturate: If False, applies `abs()` to avoid negatives.
            mode: The convolution shape.
            premultiply: Optional scalar to multiply the result before normalization.
            multiply: Optional scalar to multiply the result at the end.
            clamp: If True, clamps the final result to [RangeMin, RangeMax].

        Returns:
            A `TupleExprList` representing the expression-based convolution.

        Raises:
            CustomValueError: If matrix length is invalid or doesn't match the mode.
        """
        convolution = list[SupportsSumNoDefaultT](flatten(matrix))

        if not (conv_len := len(convolution)) % 2:
            raise CustomValueError("Convolution length must be odd!", cls.convolution, matrix)
        elif conv_len < 3:
            raise CustomValueError("You must pass at least 3 convolution items!", cls.convolution, matrix)
        elif mode == ConvMode.SQUARE and conv_len != isqrt(conv_len) ** 2:
            raise CustomValueError(
                "With square mode, convolution must represent a horizontal*vertical square (radius*radius n items)!",
                cls.convolution,
            )

        radius = conv_len // 2 if mode != ConvMode.SQUARE else isqrt(conv_len) // 2

        rel_pixels = cls.matrix(var, radius, mode)

        output = TupleExprList(
            [
                ExprList(
                    [
                        rel_pix if weight == 1 else [rel_pix, weight, cls.MUL]
                        for rel_pix, weight in zip(rel_px, convolution)
                        if weight != 0
                    ]
                )
                for rel_px in rel_pixels
            ]
        )

        for out in output:
            out.extend(cls.ADD * out.mlength)

            if premultiply is not None:
                out.append(premultiply, cls.MUL)

            if divisor is not False:
                div = sum(convolution) if divisor is True else divisor

                if div not in {0, 1}:
                    out.append(str(div), cls.DIV)

            if bias is not None:
                out.append(bias, cls.ADD)

            if not saturate:
                out.append(cls.ABS)

            if multiply is not None:
                out.append(multiply, cls.MUL)

            if clamp:
                out.append(cls.clamp(ExprToken.RangeMin, ExprToken.RangeMax))

        return output

    @staticmethod
    def _parse_planes(
        planesa: ExprVars | HoldsVideoFormatT | VideoFormatT | SupportsIndex,
        planesb: ExprVars | HoldsVideoFormatT | VideoFormatT | SupportsIndex | None,
        func: FuncExceptT,
    ) -> tuple[ExprVars, ExprVars]:
        planesa = ExprVars(planesa)
        planesb = ExprVars(planesa.stop, planesa.stop + len(planesa)) if planesb is None else ExprVars(planesb)

        if len(planesa) != len(planesb):
            raise CustomIndexError("Both clips must have an equal amount of planes!", func)

        return planesa, planesb

    @classmethod
    def rmse(
        cls,
        planesa: ExprVars | HoldsVideoFormatT | VideoFormatT | SupportsIndex,
        planesb: ExprVars | HoldsVideoFormatT | VideoFormatT | SupportsIndex | None = None,
    ) -> ExprList:
        """
        Build an expression to compute the Root Mean Squared Error (RMSE) between two plane sets.

        Args:
            planesa: The first plane set or clip.
            planesb: The second plane set or clip. If None, uses same as `planesa`.

        Returns:
            An `ExprList` representing the RMSE expression across all planes.
        """
        planesa, planesb = cls._parse_planes(planesa, planesb, cls.rmse)

        expr = ExprList()

        for a, b in zip(planesa, planesb):
            expr.append([a, b, cls.SUB, cls.DUP, cls.MUL, cls.SQRT])

        expr.append(cls.MAX * expr.mlength)

        return expr

    @classmethod
    def mae(
        cls,
        planesa: ExprVars | HoldsVideoFormatT | VideoFormatT | SupportsIndex,
        planesb: ExprVars | HoldsVideoFormatT | VideoFormatT | SupportsIndex | None = None,
    ) -> ExprList:
        """
        Build an expression to compute the Mean Absolute Error (MAE) between two plane sets.

        Args:
            planesa: The first plane set or clip.
            planesb: The second plane set or clip. If None, uses same as `planesa`.

        Returns:
            An `ExprList` representing the MAE expression across all planes.
        """
        planesa, planesb = cls._parse_planes(planesa, planesb, cls.rmse)
        expr = ExprList()

        for a, b in zip(planesa, planesb):
            expr.append([a, b, cls.SUB, cls.ABS])

        expr.append(cls.MAX * expr.mlength)

        return expr

    @classmethod
    def atan(cls, c: SupportsString = "", n: int = 10) -> ExprList:
        """
        Build an expression to compute arctangent (atan) using domain reduction.

        Args:
            c: The expression variable or string input.
            n: The number of terms to use in the Taylor series approximation.

        Returns:
            An `ExprList` representing the arctangent expression.
        """
        # Using domain reduction when |x| > 1
        expr = ExprList(
            [
                ExprList([c, cls.DUP, "__atanvar!", cls.ABS, 1, cls.GT]),
                ExprList(
                    [
                        "__atanvar@",
                        cls.SGN.convert_extra(),
                        cls.PI,
                        cls.MUL,
                        2,
                        cls.DIV,
                        1,
                        "__atanvar@",
                        cls.DIV,
                        cls.atanf("", n),
                        cls.SUB,
                    ]
                ),
                ExprList([cls.atanf("__atanvar@", n)]),
                cls.TERN,
            ]
        )

        return expr

    @classmethod
    def atanf(cls, c: SupportsString = "", n: int = 10) -> ExprList:
        """
        Approximate atan(x) using a Taylor series centered at 0.

        This is accurate for inputs in [-1, 1]. Use `atan` for full-range values.

        Args:
            c: The expression variable or string input.
            n: The number of terms in the Taylor series (min 2).

        Returns:
            An `ExprList` approximating atan(x).
        """
        # Approximation using Taylor series
        n = max(2, n)

        expr = ExprList([c, cls.DUP, "__atanfvar!"])

        for i in range(1, n):
            expr.append("__atanfvar@", 2 * i + 1, cls.POW, 2 * i + 1, cls.DIV, cls.SUB if i % 2 else cls.ADD)

        return expr

    @classmethod
    def asin(cls, c: SupportsString = "", n: int = 10) -> ExprList:
        """
        Build an expression to approximate arcsine using an identity:
            asin(x) = atan(x / sqrt(1 - x²))

        Args:
            c: The input expression variable.
            n: Number of terms to use in the internal atan approximation.

        Returns:
            An `ExprList` representing the asin(x) expression.
        """
        return cls.atan(ExprList([c, cls.DUP, cls.DUP, cls.MUL, 1, cls.SWAP, cls.SUB, cls.SQRT, cls.DIV]).to_str(), n)

    @classmethod
    def acos(cls, c: SupportsString = "", n: int = 10) -> ExprList:
        """
        Build an expression to approximate arccosine using an identity:
            acos(x) = π/2 - asin(x)

        Args:
            c: The input expression variable.
            n: Number of terms to use in the internal asin approximation.

        Returns:
            An `ExprList` representing the acos(x) expression.
        """
        return ExprList([c, "__acosvar!", cls.PI, 2, cls.DIV, cls.asin("__acosvar@", n), cls.SUB])

    @classmethod
    def masked_merge(cls, c_a: SupportsString = "", c_b: SupportsString = "", mask: SupportsString = "") -> ExprList:
        """
        Build a masked merge expression from two inputs and a mask.

        Args:
            c_a: The first input expression variable.
            c_b: The second input expression variable.
            mask: The mask expression that determines how `c_a` and `c_b` are combined.

        Returns:
            An `ExprList` representing the MaskedMerge expression.
        """
        return ExprList([c_a, c_b, [mask, ExprToken.RangeMax, ExprToken.RangeMin, cls.SUB, cls.DIV], cls.LERP])

    @classmethod
    def polyval(cls, c: SupportsString, *coeffs: SupportsString) -> ExprList:
        """
        Build an expression to evaluate a polynomial at a given value using Horner's method.

        Args:
            c: The input expression variable at which the polynomial is evaluated (the 'x' value).
            *coeffs: Coefficients of the polynomial. Must provide at least one coefficient.

        Returns:
            An `ExprList` representing the polyval expression.

        Raises:
            CustomValueError: If fewer than one coefficient is provided.
        """
        if len(coeffs) < 1:
            raise CustomValueError("You must provide at least one coefficient.", cls.polyval, coeffs)

        if b"polyval" in _get_akarin_expr_version()["expr_features"]:
            return ExprList([*coeffs, c, ExprOp.POLYVAL(len(coeffs) - 1)])

        stack_len = len(coeffs) + 1

        expr = ExprList([*coeffs, c, 0])

        for i in range(stack_len, 1, -1):
            expr.append(ExprOp.DUPN(1), ExprOp.MUL, ExprOp.DUPN(i), ExprOp.ADD)

        expr.append(ExprOp.SWAPN(stack_len), ExprOp.DROPN(stack_len))

        return expr
