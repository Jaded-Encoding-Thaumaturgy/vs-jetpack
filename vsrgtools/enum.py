from __future__ import annotations

from enum import auto
from math import ceil, exp, pi, sqrt
from typing import Any, Iterable, Literal, TypeVar, overload

from jetpytools import CustomEnum, CustomNotImplementedError, FuncExceptT, to_arr
from typing_extensions import Self

from vsexprtools import ExprList, ExprOp, ExprToken, ExprVars
from vstools import (
    ConstantFormatVideoNode,
    ConvMode,
    CustomValueError,
    KwargsT,
    PlanesT,
    check_variable_format,
    core,
    fallback,
    iterate,
    shift_clip_multi,
    vs,
)

__all__ = ["BlurMatrix", "BlurMatrixBase"]


_Nb = TypeVar("_Nb", bound=float | int)


class BlurMatrixBase(list[_Nb]):
    """
    Represents a convolution kernel (matrix) for spatial or temporal filtering.

    This class is typically constructed via the `BlurMatrix` enum, and encapsulates both the filter values
    and the intended convolution mode (e.g., horizontal, vertical, square, temporal).

    When called, it applies the convolution to a clip using the appropriate method (`std.Convolution`,
    `std.AverageFrames`, or a custom `ExprOp` expression), depending on the kernel's properties and context.

    Example:
        ```py
        kernel = BlurMatrix.BINOMIAL(taps=2)
        blurred = kernel(clip)
        ```
    """

    def __init__(self, iterable: Iterable[_Nb], /, mode: ConvMode = ConvMode.SQUARE) -> None:
        """
        Args:
            iterable: Iterable of kernel coefficients.
            mode: Convolution mode to use. Default is SQUARE.
        """
        self.mode = mode
        super().__init__(iterable)

    def __call__(
        self,
        clip: vs.VideoNode | Iterable[vs.VideoNode],
        planes: PlanesT = None,
        bias: float | None = None,
        divisor: float | None = None,
        saturate: bool = True,
        passes: int = 1,
        func: FuncExceptT | None = None,
        expr_kwargs: KwargsT | None = None,
        **conv_kwargs: Any,
    ) -> ConstantFormatVideoNode:
        """
        Apply the blur kernel to the given clip via spatial or temporal convolution.

        Chooses the appropriate backend (`std.Convolution`, `std.AverageFrames`, or `ExprOp.convolution`)
        depending on kernel size, mode, format, and other constraints.

        Args:
            clip: Source clip.
            planes: Planes to process. Defaults to all.
            bias: Value added to result before clamping.
            divisor: Divides the result of the convolution (before adding bias). Defaults to sum of kernel values.
            saturate: If True, negative values are clamped to zero. If False, absolute values are returned.
            passes: Number of convolution passes to apply.
            func: Function returned for custom error handling. This should only be set by VS package developers.
            expr_kwargs: Extra kwargs passed to ExprOp.convolution when used.
            **conv_kwargs: Any other args passed to the underlying VapourSynth function.

        Returns:
            Processed (blurred) video clip.
        """
        clip = to_arr(clip)

        func = func or self

        assert check_variable_format(clip, func)

        if len(self) <= 1:
            return clip[0]

        expr_kwargs = expr_kwargs or {}

        fp16 = clip[0].format.sample_type == vs.FLOAT and clip[0].format.bits_per_sample == 16

        # Spatial mode
        if self.mode.is_spatial:
            if len(clip) > 1:
                raise CustomValueError("You can't pass multiple clips when using a spatial mode.", func)

            # TODO: https://github.com/vapoursynth/vapoursynth/issues/1101
            if all([not fp16, len(self) <= 25, all(-1023 <= x <= 1023 for x in self), self.mode != ConvMode.SQUARE]):
                return iterate(clip[0], core.std.Convolution, passes, self, bias, divisor, planes, saturate, self.mode)

            return iterate(
                clip[0],
                ExprOp.convolution("x", self, bias, fallback(divisor, True), saturate, self.mode, **expr_kwargs),
                passes,
                planes=planes,
                **conv_kwargs,
            )

        # Temporal mode
        use_std = all(
            [
                not fp16,
                len(self) <= 31,
                all(-1023 <= x <= 1023 for x in self),
                not bias,
                saturate,
            ]
        )

        if len(clip) > 1:
            if passes != 1:
                raise CustomValueError(
                    "`passes` are not supported when passing multiple clips in temporal mode", func, passes
                )

            if use_std:
                return core.std.AverageFrames(clip, self, divisor, planes=planes)

            return ExprOp.convolution(
                ExprVars(len(clip)), self, bias, fallback(divisor, True), saturate, self.mode, **expr_kwargs
            )(clip, planes=planes, **conv_kwargs)

        # std.AverageFrames doesn't support premultiply, multiply and clamp from ExprOp.convolution
        if use_std and conv_kwargs.keys() <= {"scenechange"}:
            return iterate(clip[0], core.std.AverageFrames, passes, self, divisor, planes=planes, **conv_kwargs)

        return self._averageframes_akarin(clip[0], planes, bias, divisor, saturate, passes, expr_kwargs, **conv_kwargs)

    def _averageframes_akarin(self, *args: Any, **kwargs: Any) -> ConstantFormatVideoNode:
        clip, planes, bias, divisor, saturate, passes, expr_kwargs = args
        conv_kwargs = kwargs

        r = len(self) // 2

        if conv_kwargs.pop("scenechange", False) is False:
            expr_conv = ExprOp.convolution(
                ExprVars(len(self)), self, bias, fallback(divisor, True), saturate, self.mode, **conv_kwargs
            )
            out = iterate(clip, lambda x: expr_conv(shift_clip_multi(x, (-r, r)), planes=planes, **expr_kwargs), passes)
            return core.std.CopyFrameProps(out, clip)

        expr = ExprList()

        vars_ = [[f"{v}"] for v in ExprOp.matrix(ExprVars(len(self), akarin=True), r, self.mode)[0]]

        back_vars = vars_[:r]

        # Constructing the expression for backward (previous) clips.
        # Each clip is weighted by its corresponding weight and multiplied by the logical NOT
        # of all `_SceneChangeNext` properties from the current and subsequent clips.
        # This ensures that the expression excludes frames that follow detected scene changes.
        for i, (var, weight) in enumerate(zip(back_vars, self[:r])):
            expr.append(
                var,
                weight,
                ExprOp.MUL,
                [[f"{back_vars[ii][0]}._SceneChangeNext", ExprOp.NOT, ExprOp.MUL] for ii in range(i, len(back_vars))],
                ExprOp.DUP,
                f"cond{i}!",
            )

        forw_vars = vars_[r + 1 :]
        forw_vars.reverse()

        # Same thing for forward (next) clips.
        for j, (var, weight) in enumerate(zip(forw_vars, reversed(self[r + 1 :]))):
            expr.append(
                var,
                weight,
                ExprOp.MUL,
                [[f"{forw_vars[jj][0]}._SceneChangePrev", ExprOp.NOT, ExprOp.MUL] for jj in range(j, len(forw_vars))],
                ExprOp.DUP,
                f"cond{len(vars_) - j - 1}!",
            )

        # If a scene change is detected, all the weights beyond it are applied
        # to the center frame.
        expr.append(vars_[r], self[r])

        for k, w in enumerate([*self[:r], None, *self[r + 1 :]]):
            if w is not None:
                expr.append(f"cond{k}@", 0, w, ExprOp.TERN)

        expr.append(ExprOp.ADD * r * 2, ExprOp.MUL, ExprOp.ADD * r * 2)

        if premultiply := conv_kwargs.get("premultiply", None):
            expr.append(premultiply, ExprOp.MUL)

        if divisor:
            expr.append(divisor, ExprOp.DIV)
        else:
            expr.append(sum(self), ExprOp.DIV)

        if bias:
            expr.append(bias, ExprOp.ADD)

        if not saturate:
            expr.append(ExprOp.ABS)

        if multiply := conv_kwargs.get("multiply", None):
            expr.append(multiply, ExprOp.MUL)

        if conv_kwargs.get("clamp", False):
            expr.append(ExprOp.clamp(ExprToken.RangeMin, ExprToken.RangeMax))

        out = iterate(clip, lambda x: expr(shift_clip_multi(x, (-r, r)), planes=planes, **expr_kwargs), passes)

        return core.std.CopyFrameProps(out, clip)

    def outer(self) -> Self:
        """
        Convert a 1D kernel into a 2D square kernel by computing the outer product.

        Returns:
            New `BlurMatrixBase` instance with 2D kernel and same mode.
        """
        from numpy import outer

        return self.__class__(list[_Nb](outer(self, self).flatten()), self.mode)  # pyright: ignore[reportArgumentType]


class BlurMatrix(CustomEnum):
    """
    Enum for predefined 1D and 2D blur kernel generators.

    Provides commonly used blur kernels (e.g., mean, binomial, Gaussian) for convolution-based filtering.

    Each kernel is returned as a `BlurMatrixBase` object.
    """

    MEAN_NO_CENTER = auto()
    """
    Mean kernel excluding the center pixel. Also aliased as BOX_BLUR_NO_CENTER.
    """

    BOX_BLUR_NO_CENTER = MEAN_NO_CENTER

    MEAN = auto()
    """
    Standard mean/box blur kernel including the center pixel. Aliased as BOX_BLUR.
    """

    BOX_BLUR = MEAN

    BINOMIAL = auto()
    """
    Pascal triangle coefficients approximating Gaussian blur.
    """

    GAUSS = auto()
    """
    Proper Gaussian kernel defined by `sigma`.
    """

    @overload
    def __call__(  # type: ignore[misc]
        self: Literal[BlurMatrix.MEAN_NO_CENTER], taps: int = 1, *, mode: ConvMode = ConvMode.SQUARE
    ) -> BlurMatrixBase[int]: ...

    @overload
    def __call__(  # type: ignore[misc]
        self: Literal[BlurMatrix.MEAN], taps: int = 1, *, mode: ConvMode = ConvMode.SQUARE
    ) -> BlurMatrixBase[int]: ...

    @overload
    def __call__(  # type: ignore[misc]
        self: Literal[BlurMatrix.BINOMIAL], taps: int = 1, *, mode: ConvMode = ConvMode.HV
    ) -> BlurMatrixBase[int]: ...

    @overload
    def __call__(  # type: ignore[misc]
        self: Literal[BlurMatrix.GAUSS],
        taps: int | None = None,
        *,
        sigma: float = 0.5,
        mode: ConvMode = ConvMode.HV,
        **kwargs: Any,
    ) -> BlurMatrixBase[float]: ...

    @overload
    def __call__(self, taps: int | None = None, **kwargs: Any) -> Any: ...

    def __call__(self, taps: int | None = None, **kwargs: Any) -> Any:
        """
        Generate the blur kernel based on the enum variant.

        Args:
            taps: Size of the kernel in each direction.
            sigma: [GAUSS only] Standard deviation of the Gaussian kernel.
            mode: Convolution mode. Default depends on kernel.

        Returns:
            A `BlurMatrixBase` instance representing the kernel.
        """
        kernel: BlurMatrixBase[Any]

        match self:
            case BlurMatrix.MEAN_NO_CENTER:
                taps = fallback(taps, 1)
                mode = kwargs.pop("mode", ConvMode.SQUARE)

                matrix = [1 for _ in range(((2 * taps + 1) ** (2 if mode == ConvMode.SQUARE else 1)) - 1)]
                matrix.insert(len(matrix) // 2, 0)

                return self.custom(matrix, mode)

            case BlurMatrix.MEAN:
                taps = fallback(taps, 1)
                mode = kwargs.pop("mode", ConvMode.SQUARE)

                kernel = self.custom((1 for _ in range((2 * taps + 1))), mode)

            case BlurMatrix.BINOMIAL:
                taps = fallback(taps, 1)
                mode = kwargs.pop("mode", ConvMode.HV)

                c = 1
                n = taps * 2 + 1

                matrix = list[int]()

                for i in range(1, taps + 2):
                    matrix.append(c)
                    c = c * (n - i) // i

                kernel = self.custom(matrix[:-1] + matrix[::-1], mode)

            case BlurMatrix.GAUSS:
                taps = fallback(taps, 1)
                sigma = kwargs.pop("sigma", 0.5)
                mode = kwargs.pop("mode", ConvMode.HV)
                scale_value = kwargs.pop("scale_value", 1023)

                if mode == ConvMode.SQUARE:
                    scale_value = sqrt(scale_value)

                taps = self.get_taps(sigma, taps)

                if taps < 0:
                    raise CustomValueError("Taps must be >= 0!")

                if sigma > 0.0:
                    half_pisqrt = 1.0 / sqrt(2.0 * pi) * sigma
                    doub_qsigma = 2 * sigma**2

                    high, *mat = [half_pisqrt * exp(-(x**2) / doub_qsigma) for x in range(taps + 1)]

                    mat = [x * scale_value / high for x in mat]
                    mat = [*mat[::-1], scale_value, *mat]
                else:
                    mat = [scale_value]

                kernel = self.custom(mat, mode)

            case _:
                raise CustomNotImplementedError("Unsupported blur matrix enum!", self, self)

        if mode == ConvMode.SQUARE:
            kernel = kernel.outer()

        return kernel

    def from_radius(self: Literal[BlurMatrix.GAUSS], radius: int) -> BlurMatrixBase[float]:  # type: ignore[misc]
        """
        Generate a Gaussian blur kernel from an intuitive radius.

        This is a shortcut that converts a blur radius to a corresponding sigma value.

        Args:
            radius: Blur radius.

        Returns:
            Gaussian blur matrix.
        """
        assert self is BlurMatrix.GAUSS

        return BlurMatrix.GAUSS(None, sigma=(radius + 1.0) / 3)

    def get_taps(self: Literal[BlurMatrix.GAUSS], sigma: float, taps: int | None = None) -> int:  # type: ignore[misc]
        """
        Compute the number of taps required for a given sigma value.

        Args:
            sigma: Gaussian sigma value.
            taps: Optional manual override; if not provided, it's computed from sigma.

        Returns:
            Number of taps.
        """
        assert self is BlurMatrix.GAUSS

        if taps is None:
            taps = ceil(abs(sigma) * 8 + 1) // 2

        return taps

    @classmethod
    def custom(cls, values: Iterable[_Nb], mode: ConvMode = ConvMode.SQUARE) -> BlurMatrixBase[_Nb]:
        """
        Create a custom BlurMatrixBase kernel with explicit values and mode.

        Args:
            values: The kernel coefficients.
            mode: Convolution mode to use.

        Returns:
            A BlurMatrixBase instance.
        """
        return BlurMatrixBase(values, mode=mode)
