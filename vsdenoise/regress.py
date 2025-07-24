from __future__ import annotations

# ruff: noqa

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Concatenate, Sequence

from vsaa import EEDI3, NNEDI3, SangNom
from vsexprtools import ExprOp, norm_expr
from vskernels import Catrom, Kernel, KernelLike, Point, Scaler, ScalerLike
from vsrgtools import box_blur, gauss_blur, limit_filter
from vsscale import ScalingArgs
from vstools import (
    P1,
    CustomIntEnum,
    CustomOverflowError,
    CustomStrEnum,
    FuncExceptT,
    InvalidColorFamilyError,
    InvalidSubsamplingError,
    P,
    VSFunction,
    complex_hash,
    depth,
    flatten,
    get_plane_sizes,
    get_subsampling,
    inject_self,
    join,
    split,
    vs,
    vs_object,
)
import contextlib

__all__ = [
    "ChromaReconstruct",
    "GenericChromaRecon",
    "MissingFieldsChromaRecon",
    "PAWorksChromaRecon",
    "Point422ChromaRecon",
    "ReconDiffMode",
    "ReconOutput",
    "Regression",
]


class _CachedBlurs(vs_object, dict[int, list[tuple[vs.VideoNode, vs.VideoNode]]]):
    def __vs_del__(self, core_id: int) -> None:
        self.clear()


_cached_blurs = _CachedBlurs()


@dataclass
class Regression:
    """
    Class for math operation on a clip.

    For more info see `this Wikipedia article <https://en.wikipedia.org/wiki/Regression_analysis>`_.
    """

    @dataclass
    class Linear:
        """
        Representation of a Linear Regression.

        For more info see `this Wikipedia article <https://en.wikipedia.org/wiki/Linear_regression>`_.
        """

        slope: vs.VideoNode
        """
        One of the regression coefficients.

        In simple linear regression the coefficient is the regression slope.
        """

        intercept: vs.VideoNode
        """Component of slope, the intercept term."""

        correlation: vs.VideoNode
        """The relationship between the error term and the regressors."""

    class BlurConf:
        """
        Class for the blur (or averaging filter) used for regression.
        """

        def __init__(
            self, func: Callable[Concatenate[vs.VideoNode, P], vs.VideoNode], /, *args: P.args, **kwargs: P.kwargs
        ) -> None:
            """
            Args:
                func: Function used for blurring.
                *args: Positional arguments passed to the function.
                **kwargs: Keyword arguments passed to the function.
            """

            self.func = func
            self.args = args
            self.kwargs = kwargs

        @classmethod
        def from_param(
            cls,
            func: Callable[Concatenate[vs.VideoNode, P1], vs.VideoNode] | Regression.BlurConf,
            *args: P1.args,
            **kwargs: P1.kwargs,
        ) -> Regression.BlurConf:
            """
            Get a BlurConf from generic parameters.

            Args:
                func: Function used for blurring or already existing config.
                *args: Positional arguments passed to the function.
                **kwargs: Keyword arguments passed to the function.

            Returns:
                BlurConf object.
            """

            if isinstance(func, Regression.BlurConf):
                return func.extend(*args, **kwargs)

            return Regression.BlurConf(func, *args, **kwargs)

        def extend(self, *args: Any, **kwargs: Any) -> Regression.BlurConf:
            """
            Extend the current config arguments and get a new BlurConf object.

            Args:
                *args: Positional arguments passed to the function.
                **kwargs: Keyword arguments passed to the function.

            Returns:
                BlurConf object.
            """
            if args or kwargs:
                return Regression.BlurConf(
                    self.func,
                    *(args or self.args),
                    **(self.kwargs | kwargs),  # type: ignore[arg-type]
                )
            return self

        def __call__(self, clip: vs.VideoNode, chroma_only: bool = False, *args: Any, **kwargs: Any) -> vs.VideoNode:
            """
            Blur a clip with the current config.

            Args:
                clip: Clip to be blurred.
                chroma_only: Try only processing chroma.
                *args: Positional arguments passed to the function.
                **kwargs: Keyword arguments passed to the function.

            Returns:
                Blurred clip.
            """

            if not args:
                args = self.args

            kwargs = self.kwargs | kwargs

            out = None

            if chroma_only:
                ckwargs = kwargs | {"planes": [1, 2]}

                key = complex_hash.hash(args, ckwargs)

                got_result = _cached_blurs.get(key, None)

                if got_result is not None:
                    for inc, outc in got_result:
                        if inc == clip:
                            return outc

                with contextlib.suppress(Exception):
                    out = self.func(clip, *args, **ckwargs)  # type: ignore[arg-type]

            if not out:
                key = complex_hash.hash(args, kwargs)

                got_result = _cached_blurs.get(key, None)

                if got_result is not None:
                    for inc, outc in got_result:
                        if inc == clip:
                            return outc

                out = self.func(clip, *args, **kwargs)  # type: ignore[arg-type]

            if key not in _cached_blurs:
                _cached_blurs[key] = []

            _cached_blurs[key].append((clip, out))

            return out

        def blur(self, clip: vs.VideoNode, chroma_only: bool = False, *args: Any, **kwargs: Any) -> Any:
            """
            Blur a clip with the current config.

            Args:
                clip: Clip to be blurred.
                chroma_only: Try only processing chroma.
                *args: Positional arguments passed to the function.
                **kwargs: Keyword arguments passed to the function.

            Returns:
                Blurred clip.
            """

            return self(clip, chroma_only, *args, **kwargs)

        def get_bases(
            self, clip: vs.VideoNode | Sequence[vs.VideoNode]
        ) -> tuple[Sequence[vs.VideoNode], Sequence[vs.VideoNode], Sequence[vs.VideoNode]]:
            """
            Get the base elements for a regression.

            Args:
                clip: Clip or individual planes to be processed.

            Returns:
                Tuple containing the blurred clips, variations, and relation of the two.
            """

            planes = clip if isinstance(clip, Sequence) else split(clip)

            blur = [self(shifted) for shifted in planes]

            variation = [
                norm_expr([Ex, self(ExprOp.MUL.combine(shifted, suffix=ExprOp.DUP))], "y x dup * - 0 max", func=self)
                for Ex, shifted in zip(blur, planes)
            ]

            var_mul = [self(ExprOp.MUL.combine(planes[0], shifted_y)) for shifted_y in planes[1:]]

            return blur, variation, var_mul

    blur_func: BlurConf | VSFunction[vs.VideoNode] = BlurConf(box_blur, radius=2)
    """Function used for blurring (averaging)."""

    eps: float = 1e-7
    """Epsilon, used in expressions to avoid division by zero."""

    def __post_init__(self) -> None:
        self.blur_conf = Regression.BlurConf.from_param(self.blur_func)

    @classmethod
    def from_param(
        cls,
        func: Callable[Concatenate[vs.VideoNode, P1], vs.VideoNode] | Regression.BlurConf,
        *args: P1.args,
        **kwargs: P1.kwargs,
    ) -> Regression:
        """
        Get a Regression from generic parameters.

        Args:
            func: Function used for blurring or a preconfigured BlurConf.
            *args: Positional arguments passed to the blurring function.
            **kwargs: Keyword arguments passed to the blurring function.

        Returns:
            Regression object.
        """

        return Regression(Regression.BlurConf.from_param(func, *args, **kwargs))

    def linear(
        self,
        clip: vs.VideoNode | Sequence[vs.VideoNode],
        weight: float = 0.0,
        intercept_scale: float = 50.0,
        *args: Any,
        **kwargs: Any,
    ) -> list[Regression.Linear]:
        """
        Perform a simple linear regression.

        Args:
            clip: Clip or singular planes to be processed.
            *args: Positional arguments passed to the blurring function.
            **kwargs: Keyword arguments passed to the blurring function.

        Returns:
            List of a Regression.Linear object for each plane.
        """

        blur_conf = self.blur_conf.extend(*args, **kwargs)

        (blur_x, *blur_ys), (var_x, *var_ys), var_mul = blur_conf.get_bases(clip)

        if weight < 0.0 or weight >= 1.0:
            raise CustomOverflowError(
                '"weight" must be between 0.0 and 1.0 (exclusive)!', self.__class__.linear, weight
            )

        cov_xys = [
            norm_expr([vm_y, blur_x, Ey], "x y z * -", func=self.__class__.linear) for vm_y, Ey in zip(var_mul, blur_ys)
        ]

        slopes = [norm_expr([cov_xy, var_x], f"x y {self.eps} + /", func=self.__class__.linear) for cov_xy in cov_xys]

        scale_str = f"{intercept_scale} /" if intercept_scale != 0 else ""
        intercepts = [
            norm_expr([blur_y, slope, blur_x], f"x y z * - {scale_str}", func=self.__class__.linear)
            for blur_y, slope in zip(blur_ys, slopes)
        ]

        weight_str = f"{1 - weight} - {weight} / dup 0 > swap 0 ?" if weight > 0.0 else ""

        corrs = [
            norm_expr(
                [cov_xy, var_x, var_y], f"x dup * y z * {self.eps} + / sqrt {weight_str}", func=self.__class__.linear
            )
            for cov_xy, var_y in zip(cov_xys, var_ys)
        ]

        return [
            Regression.Linear(slope, intercept, correlation)
            for slope, intercept, correlation in zip(slopes, intercepts, corrs)
        ]

    def sloped_corr(
        self,
        clip: vs.VideoNode | Sequence[vs.VideoNode],
        weight: float = 0.5,
        avg: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Sequence[vs.VideoNode]:
        """
        Compute correlation of slopes of a simple regression.

        Args:
            clip: Clip or individual planes to be processed.
            avg: Average (blur) the final result.
            *args: Positional arguments passed to the blurring function.
            **kwargs: Keyword arguments passed to the blurring function.

        Returns:
            List of clips representing the correlation of slopes.
        """

        blur_conf = self.blur_conf.extend(*args, **kwargs)

        (blur_x, *blur_ys), (var_x, *var_ys), var_mul = blur_conf.get_bases(clip)

        if weight < 0.0 or weight >= 1.0:
            raise CustomOverflowError(
                '"weight" must be between 0.0 and 1.0 (exclusive)!', self.__class__.sloped_corr, weight
            )

        coeff_x, coeff_y = weight, 1.0 - weight

        weight_str = f"{coeff_x} - {coeff_y} / 0 max" if coeff_x else ""

        corr_slopes = [
            norm_expr(
                [Exys_y, blur_x, Ex_y, var_x, var_y],
                f"x y z * - XYS! XYS@ a {self.eps} + / XYS@ dup * a b * {self.eps} + / sqrt {weight_str} *",
                func=self.__class__.sloped_corr,
            )
            if True  # complexpr_available
            else norm_expr(
                [norm_expr([Exys_y, blur_x, Ex_y], "x y z * -"), var_x, var_y],
                f"x y {self.eps} + / x dup * y z * {self.eps} + / sqrt {weight_str} *",
                func=self.__class__.sloped_corr,
            )
            for Exys_y, Ex_y, var_y in zip(var_mul, blur_ys, var_ys)
        ]

        if not avg:
            return corr_slopes

        return [blur_conf(corr_slope) for corr_slope in corr_slopes]


class ReconOutput(CustomIntEnum):
    """
    Enum to decide what combination of luma-chroma to output in ``ChromaReconstruct``
    """

    NATIVE = 0
    """
    Return 4:4:4 with luma from ``get_base_clip`` and reconstructed chroma.
    If for example your anime is native 720p, it will output the
    descaled luma in ``get_base_clip`` with 720p reconstructed chroma.
    """

    i420 = 1
    """
    Return 4:2:0 chroma as per input clip and reconstructed chroma downscaled/upscaled to fit the subsampling.
    """

    i444 = 2
    """
    Return 4:4:4 chroma as per input clip and reconstructed chroma downscaled/upscaled to fit the subsampling.
    """

    @classmethod
    def from_param(cls, value: int | ReconOutput | bool | None, func_except: FuncExceptT | None = None) -> ReconOutput:
        if isinstance(value, bool):
            value = 1 + int(value)
        elif value is None:
            return cls.NATIVE

        return super().from_param(value, func_except)  # type: ignore


@dataclass
class ReconDiffModeConf:
    """
    Internal structure.
    """

    mode: ReconDiffMode
    diff_sigma: float
    inter_scale: float


class ReconDiffMode(CustomStrEnum):
    """Enum for configuring a reconstruction difference mode."""

    SIMPLE = "x y +"
    """
    Simple demangled chroma + regressed diff merge. It is the most simple merge available.
    """

    BOOSTX = "x z * y +"
    """
    Demangled chroma * luma diff + regressed diff merge. Pay attention to overshoot.
    """

    BOOSTY = "x y z * +"
    """
    Demangled chroma + regressed diff * luma diff merge. Pay attention to overshoot.
    """

    MEAN = f"{SIMPLE} x z * y z / + + 2 /"
    """
    Simple mean of ``SIMPLE``, ``BOOSTX``, and ``BOOSTY``. Will give a dampened output.
    """

    MEDIAN = f"{MEAN} AX! {BOOSTX} BX! {BOOSTY} CX! a BX@ - abs BD! a AX@ - abs BD@ < AX@ BD@ a CX@ - abs > BX@ CX@ ? ?"
    """
    The most complex merge available, combining all other modes while still
    avoiding overshoots and undershoots while retaining the sharpness.
    """

    def __call__(self, diff_sigma: float = 0.5, inter_scale: float = 0.0) -> ReconDiffModeConf:
        """
        Configure the current mode. **It will not have any effect with ``SIMPLE``.

        Args:
            diff_sigma: Gaussian blur sigma for the luma-mangled luma difference.
            inter_scale: Scaling for using the luma-chroma difference intercept.

                   - ``= 0.0``   => Disable usage of intercept.
                   - ``< 20.0``  => Will amplify and overshoot/undershoot all bright/dark spots. Not recommended.
                   - ``< 50.0``  => Will dampen haloing and normalize chroma to luma, removing eventual bleeding.
                   - ``> 100.0`` => Placebo effect.

        Returns:
            Configured mode.
        """
        return ReconDiffModeConf(self, diff_sigma, inter_scale)


@dataclass
class ChromaReconstruct(ABC):
    """
    Class to ease the creation and usage of chroma reconstruction
    based on linear regression between luma-demangled luma and chroma-demangled chroma.

    The reconstruction depends on the following plugin:
        - https://github.com/Jaded-Encoding-Thaumaturgy/vapoursynth-reconstruct
    """

    kernel: KernelLike = field(default=Catrom, kw_only=True)
    """Base kernel used to shift/scale luma and chroma planes."""

    scaler: ScalerLike | None = field(default=None, kw_only=True)
    """Base kernel used to shift/scale luma and chroma planes."""

    _default_diff_sigma: ClassVar[float] = 0.5
    _default_inter_scale: ClassVar[float] = 0.0

    def __post_init__(self) -> None:
        self._kernel = Kernel.ensure_obj(self.kernel)
        self._scaler = Scaler.ensure_obj(self.scaler or self._kernel)

    @abstractmethod
    def get_base_clip(self, clip: vs.VideoNode) -> vs.VideoNode:
        """
        Get the base clip on which the linear regression will be applied to.

        Needs to be the native resolution the content was produced at.
        Additionally, chroma needs to be scaled to 444 for later comparison
        and overshoot/undershoot protection.

        For example, if the anime is 720p native, this function needs to output 720p 4:4:4.
        Later, chroma will be upscaled at this resolution maximum and
        will be upscaled/downscaled to 420/444 based on ``out_mode`` in ``reconstruct``.
        """

    @abstractmethod
    def get_mangled_luma(self, clip: vs.VideoNode, y_base: vs.VideoNode) -> vs.VideoNode:
        """
        Return the mangled luma to the base resolution of the content.

        Chroma might have been further mangled or can be better demangled,
        but this method assumes that the luma will be taken as the same resolution as the INPUT clip.

        So, for example, at 1080p 4:2:0 this method should return mangled luma like chroma was at 960x540.
        EVEN IF the native resolution is lower.
        """

    @abstractmethod
    def demangle_luma(self, mangled: vs.VideoNode, y_base: vs.VideoNode) -> vs.VideoNode:
        """
        Return the demangled luma. You may use the y_base to limit the damage that was done in ``get_mangled_luma``
        but it is important that some artifacting from demangling chroma in ``demangle_chroma`` remains.

        May it be blurring or the interpolator artifacts (like SangNom random bright/dark pixels).

        Assumes that the resolutions matches ``y_base``.
        """

    @abstractmethod
    def demangle_chroma(self, mangled: vs.VideoNode, y_base: vs.VideoNode) -> vs.VideoNode:
        """
        Return the demangled luma as best quality as you can.

        Assumes that the resolutions matches ``y_base``.
        """

    def get_chroma_shift(self, y_width: int, c_width: int) -> float:
        return 0.5 * c_width / y_width

    def _get_bases(
        self, clip: vs.VideoNode, include_edges: bool, func: FuncExceptT
    ) -> tuple[vs.VideoNode, vs.VideoNode, vs.VideoNode, vs.VideoNode, Sequence[vs.VideoNode], Sequence[vs.VideoNode]]:
        InvalidColorFamilyError.check(clip, vs.YUV, func)

        clip32 = depth(clip, 32)

        y, *chroma = split(clip32)

        base = self.get_base_clip(clip32)

        if get_subsampling(base) != "444":
            raise InvalidSubsamplingError(self.__class__, base, "``get_base_clip`` should return a YUV444 clip!")

        y_base, *chroma_base = split(base)

        y_m = self.get_mangled_luma(clip32, y_base)

        y_dm = self.demangle_luma(y_m, y_base)

        if include_edges:
            y_dm = self._kernel.shift(y_dm, 0.125, 0.125)

        chroma_dm = [self.demangle_chroma(x, y_base) for x in chroma]

        return y, y_base, y_m, y_dm, chroma_base, chroma_dm

    @inject_self.init_kwargs
    def debug(self, clip: vs.VideoNode, *args: Any, **kwargs: Any) -> tuple[vs.VideoNode, ...]:
        """
        In 'debug' mode you can see the various steps of mangled and demangled planes.

        Useful to determine if shifts and the sort are correct.

        The *args, **kwargs don't do anything and are there just to be able to
        hotswap reconstruct with this method without removing other arguments.
        """
        y, y_base, y_m, y_dm, chroma_base, chroma_dm = self._get_bases(clip, False, self.debug)

        return y, y_base, y_dm, *flatten(zip(chroma_base, chroma_dm))

    @inject_self.init_kwargs
    def reconstruct(
        self,
        clip: vs.VideoNode,
        sigma: float,
        radius: int,
        diff_mode: ReconDiffMode | ReconDiffModeConf,
        out_mode: ReconOutput | bool | None,
        include_edges: bool,
        lin_cutoff: float = 0.0,
        **kwargs: Any,
    ) -> vs.VideoNode:
        """
        Run the actual reconstructing implemented in this class.

        Args:
            clip: Input clip. Must be YUV.
            sigma: Sigma for gaussian blur of weights, higher value is useful to dampen wrong directions.
            radius: Radius of the reconstruct window. Higher will be more stable but also less sharp and will adhere
                less to luma.
            diff_mode: The mode to apply the difference to apply, calculated with linear regression, to the mangled
                chroma. Check ``ReconDiffMode`` to know what each mode means.
            out_mode: The luma/chroma output combination.
            include_edges: Forcecully include all luma edges in the weighting.
            lin_cutoff: Cutoff, or weight, in the linear regression.

        Returns:
            Clip with demangled chroma.
        """

        y, y_base, y_m, y_dm, chroma_base, chroma_dm = self._get_bases(clip, include_edges, self.reconstruct)

        reg = Regression.from_param(Regression.BlurConf(gauss_blur, sigma=sigma))  # pyright: ignore

        if not isinstance(diff_mode, ReconDiffModeConf):
            diff_mode = diff_mode()

        chroma_regs = reg.linear([y_dm, *chroma_dm], lin_cutoff, diff_mode.inter_scale)

        y_diff = norm_expr((y_base, y_dm), "x y -", func=self.reconstruct)

        y_diffxb = gauss_blur(
            norm_expr((y_base, y_dm), f"x y / {reg.eps} 1 clamp"), diff_mode.diff_sigma, func=self.reconstruct
        )

        fixup = (
            y_diff.recon.Reconstruct(  # type: ignore
                reg.slope,
                reg.correlation,
                radius=radius,
                intercept=(None if diff_mode.inter_scale == 0.0 else reg.intercept),
            )
            for reg in chroma_regs
        )

        fixed_chroma = (
            norm_expr((dm, fix, y_diffxb, base), diff_mode.mode.value, func=self.reconstruct)
            for dm, fix, base in zip(chroma_dm, fixup, chroma_base)
        )

        out_mode = ReconOutput.from_param(out_mode)

        top_shift = left_shift = 0.0

        if out_mode == ReconOutput.i420:
            left_shift = -self.get_chroma_shift(y.width, y_m.height)
        elif include_edges:
            top_shift = left_shift = 0.125 / 2

        shifted_chroma = (self._kernel.shift(p, (top_shift, left_shift)) for p in fixed_chroma)

        if out_mode != ReconOutput.NATIVE:
            y_base, targ_sizes = y, (clip.width, clip.height)

            if out_mode == ReconOutput.i420:
                targ_sizes = tuple[int, int](targ_size // 2 for targ_size in targ_sizes)  # type: ignore

            shifted_chroma = (self._scaler.scale(p, *targ_sizes) for p in shifted_chroma)

        return depth(join(y_base, *shifted_chroma), clip)


@dataclass
class GenericChromaRecon(ChromaReconstruct):
    """
    Generic ChromaReconstruct which implements base functions.

    Not reccomended to use without customizing the mangling/demangling.
    """

    native_res: int | float | None = None
    """Native resolution of the show."""

    native_kernel: KernelLike = Catrom
    """Native kernel of the show."""

    src_left: float = field(default=0.5, kw_only=True)
    """Base left shift of the interpolator. If using base vsaa scaler, this will be interally compensated."""

    src_top: float = field(default=0.0, kw_only=True)
    """Base top shift of the interpolator."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self._native_kernel = Kernel.ensure_obj(self.native_kernel, self.__class__)

    def get_base_clip(self, clip: vs.VideoNode) -> vs.VideoNode:
        if self.native_res is None:
            return self._kernel.resample(clip, vs.YUV444PS)

        de_args = ScalingArgs.from_args(clip, self.native_res)

        descale = self._native_kernel.descale(clip, de_args.width, de_args.height, **de_args.kwargs())

        return join(
            self._kernel.shift(descale, de_args.src_top / 2, -de_args.src_left / 2),
            self._scaler.scale(clip, de_args.width, de_args.height, format=vs.YUV444PS),
        )

    def get_mangled_luma(self, clip: vs.VideoNode, y_base: vs.VideoNode) -> vs.VideoNode:
        c_width, c_height = get_plane_sizes(clip, 1)

        return Catrom().scale(y_base, c_width, c_height, (0, -0.5 + self.get_chroma_shift(clip.width, c_width)))

    def demangle_chroma(self, mangled: vs.VideoNode, y_base: vs.VideoNode) -> vs.VideoNode:
        return self._kernel.scale(
            mangled, y_base.width, y_base.height, (0, self.get_chroma_shift(y_base.width, mangled.width))
        )

    def demangle_luma(self, mangled: vs.VideoNode, y_base: vs.VideoNode) -> vs.VideoNode:
        src_left, self.src_left = self.src_left, self.src_left - 0.25
        luma = self.demangle_chroma(mangled, y_base)
        self.src_left = src_left
        return luma

    @inject_self.init_kwargs
    def reconstruct(
        self,
        clip: vs.VideoNode,
        sigma: float = 1.5,
        radius: int = 2,
        diff_mode: ReconDiffMode | ReconDiffModeConf = ReconDiffMode.MEAN,
        out_mode: ReconOutput | bool | None = ReconOutput.i420,
        include_edges: bool = False,
        lin_cutoff: float = 0.0,
        **kwargs: Any,
    ) -> vs.VideoNode:
        return super().reconstruct(clip, sigma, radius, diff_mode, out_mode, include_edges, lin_cutoff)


@dataclass
class MissingFieldsChromaRecon(GenericChromaRecon):
    """
    Base helper function for reconstructing chroma with missing fields.
    """

    dm_wscaler: ScalerLike = NNEDI3
    """Scaler used to interpolate the width/height."""

    dm_hscaler: ScalerLike | None = NNEDI3
    """Scaler used to interpolate the height."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self._dm_wscaler = Scaler.ensure_obj(self.dm_wscaler, self.__class__)
        self._dm_hscaler = self._dm_wscaler.ensure_obj(self.dm_hscaler, self.__class__)


@dataclass
class PAWorksChromaRecon(MissingFieldsChromaRecon):
    """
    Chroma reconstructor for 720p PAWorks chroma which undergoes through the following mangling process:

    Produced at 720p 4:4:4
        => 720p 4:2:2 => 720p 4:4:4
            With Point, so the width gets halved fields, and the lowest it got is 640x720
        => 1080p 4:4:4 => 1080p 4:2:2 => 1080p 4:2:0
            With Catrom, so the width doesn't get affected, but gets downscaled to 960x540

    Through this process, we know the lowest the chroma was is 640x540.
    640 width from point 4:2:2 and 540 height from catrom 4:2:0.

    With this information we can implement this demangler as follows:
        - get_base_clip:
            descaled luma to 720p, upscaled chroma to 720p
        - get_mangled_luma:
            scale the descale to 620x720 (4:2:2 at 720p), then reupscale to 960x720 (4:4:4 at 720p)
            - thus, removing fields information -, then downscale the height to 540p. (4:2:0 at 1080p)
        - demangle_luma/demangle_chroma:
            downscale with point from 960x540 to 640x540 which was the lowest it got, to
            remove point interpolated fields, then reupscale.

            In the case of luma, we also limit the mangling by clamping the difference of the
            demanglers to the original descaled luma or details would just get crushed.

    """

    dm_wscaler: ScalerLike = field(default_factory=lambda: SangNom(128))
    dm_hscaler: ScalerLike = NNEDI3

    def get_mangled_luma(self, clip: vs.VideoNode, y_base: vs.VideoNode) -> vs.VideoNode:
        cm_width, _ = get_plane_sizes(y_base, 1)
        c_width, c_height = get_plane_sizes(clip, 1)

        point = Point()

        y_m = point.scale(y_base, cm_width // 2, y_base.height, (0, -1))
        y_m = point.scale(y_m, c_width, y_base.height, (0, -0.25))
        y_m = Catrom().scale(y_m, c_width, c_height)

        return y_m

    def demangle_chroma(self, mangled: vs.VideoNode, y_base: vs.VideoNode) -> vs.VideoNode:
        demangled = vs.core.resize.Point(mangled, y_base.width // 2, mangled.height)

        demangled = self._dm_wscaler.scale(demangled, mangled.width, y_base.height, (self.src_top, 0))
        demangled = self._dm_hscaler.scale(demangled, y_base.width, y_base.height, (0, self.src_left))

        return demangled

    def demangle_luma(self, mangled: vs.VideoNode, y_base: vs.VideoNode) -> vs.VideoNode:
        a = self.demangle_chroma(mangled, y_base)

        y_base = self._kernel.shift(y_base, self.src_top, self.src_left)

        return limit_filter(a, y_base, a, dark_thr=1, bright_thr=10, elast=4.5)

    @inject_self.init_kwargs
    def reconstruct(
        self,
        clip: vs.VideoNode,
        sigma: float = 2.0,
        radius: int = 4,
        diff_mode: ReconDiffMode | ReconDiffModeConf = ReconDiffMode.MEDIAN,
        out_mode: ReconOutput | bool | None = ReconOutput.NATIVE,
        include_edges: bool = True,
        lin_cutoff: float = 0.0,
        **kwargs: Any,
    ) -> vs.VideoNode:
        return super().reconstruct(clip, sigma, radius, diff_mode, out_mode, include_edges, lin_cutoff)


@dataclass
class Point422ChromaRecon(MissingFieldsChromaRecon):
    """
    Demangler for content that has undergone from 4:4:4 => 4:2:2 with point, then 4:2:0 with some neutral scaler.
    """

    dm_wscaler: ScalerLike = field(default_factory=lambda: SangNom(128))
    dm_hscaler: ScalerLike = field(
        # sclip=NNEDI3() didn't work before and has not been re-implemented either
        default_factory=lambda: EEDI3(0.35, 0.55, 20, 2, 10, vcheck=3, sclip=NNEDI3())  # type: ignore
    )

    def demangle_chroma(self, mangled: vs.VideoNode, y_base: vs.VideoNode) -> vs.VideoNode:
        demangled = self._dm_hscaler.scale(mangled, mangled.width, y_base.height)
        return self._dm_wscaler.scale(demangled, y_base.width, y_base.height, (self.src_top, self.src_left))

    @inject_self.init_kwargs
    def reconstruct(
        self,
        clip: vs.VideoNode,
        sigma: float = 1.5,
        radius: int = 2,
        diff_mode: ReconDiffMode | ReconDiffModeConf = ReconDiffMode.MEDIAN,
        out_mode: ReconOutput | bool | None = ReconOutput.i444,
        include_edges: bool = True,
        lin_cutoff: float = 0.0,
        **kwargs: Any,
    ) -> vs.VideoNode:
        return super().reconstruct(clip, sigma, radius, diff_mode, out_mode, include_edges, lin_cutoff)
