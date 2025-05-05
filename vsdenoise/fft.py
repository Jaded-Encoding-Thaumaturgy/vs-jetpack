from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Literal, Mapping, Sequence, TypeAlias, Union, overload

from jetpytools import KwargsNotNone, classproperty, fallback
from typing_extensions import Self, deprecated
from vapoursynth import Plugin

from vstools import (
    ConstantFormatVideoNode, CustomEnum, CustomIntEnum, CustomOverflowError, CustomRuntimeError, CustomValueError,
    FieldBased, FuncExceptT, PlanesT, SupportsFloatOrIndex, check_progressive, core, flatten, get_depth,
    get_sample_type, inject_self, vs
)

__all__ = [
    'DFTTest',

    'SLocationT',

    'fft3d'
]


class DFTTest:
    """2D/3D frequency domain denoiser."""

    class SLocation(Iterable[float]):
        """Specify a range of frequencies to target."""

        class InterMode(CustomEnum):
            """SLocation interpolation mode."""

            LINEAR = 'linear'
            SPLINE = 1
            SPLINE_LINEAR = 'slinear'
            QUADRATIC = 'quadratic'
            CUBIC = 'cubic'
            NEAREST = 'nearest'
            NEAREST_UP = 'nearest-up'
            ZERO = 'zero'

            @overload
            def __call__(self, location: SLocationT, /, *, res: int = 20, digits: int = 3) -> DFTTest.SLocation:
                ...

            @overload
            def __call__(self, location: SLocationT | None, /, *, res: int = 20, digits: int = 3) -> DFTTest.SLocation | None:
                ...

            @overload
            def __call__(
                self,
                h_loc: SLocationT | None,
                v_loc: SLocationT | None,
                t_loc: SLocationT | None,
                /,
                *,
                res: int = 20, digits: int = 3
            ) -> DFTTest.SLocation.MultiDim:
                ...

            def __call__(
                self, *locations: SLocationT | None, res: int = 20, digits: int = 3
            ) -> DFTTest.SLocation | None | DFTTest.SLocation.MultiDim:
                if len(locations) == 1:
                    sloc = DFTTest.SLocation.from_param(locations[0])

                    if sloc is not None:
                        sloc = sloc.interpolate(self, res, digits)

                    return sloc

                return DFTTest.SLocation.MultiDim(*(self(x, res=res, digits=digits) for x in locations))

        frequencies: tuple[float, ...]
        sigmas: tuple[float, ...]

        def __init__(
            self,
            locations: Sequence[Frequency | Sigma] | Sequence[tuple[Frequency, Sigma]] | Mapping[Frequency, Sigma],
            interpolate: InterMode | None = None,
            strict: bool = True
        ) -> None:
            if isinstance(locations, Mapping):
                frequencies, sigmas = list(locations.keys()), list(locations.values())
            else:
                locations = list[float](flatten(locations))

                if len(locations) % 2:
                    raise CustomValueError(
                        "slocations must resolve to an even number of items, pairing frequency and sigma respectively",
                        self.__class__
                    )

                frequencies, sigmas = list(locations[0::2]), list(locations[1::2])

            frequencies = self.boundsCheck(frequencies, (0, 1), strict)
            sigmas = self.boundsCheck(sigmas, (0, None), strict)

            self.frequencies, self.sigmas = (t for t in zip(*sorted(zip(frequencies, sigmas))))

            if interpolate:
                interpolated = self.interpolate(interpolate)

                self.frequencies, self.sigmas = interpolated.frequencies, interpolated.sigmas

        def __iter__(self) -> Iterator[float]:
            return iter([v for pair in zip(self.frequencies, self.sigmas) for v in pair])

        def __reversed__(self) -> Self:
            return self.__class__(
                dict(zip((1 - f for f in reversed(self.frequencies)), list(reversed(self.sigmas))))
            )

        @classmethod
        def boundsCheck(
            cls, values: list[float], bounds: tuple[float | None, float | None], strict: bool = False
        ) -> list[float]:
            if not values:
                raise CustomValueError('"values" can\'t be empty!', cls)

            values = values.copy()

            bounds_str = iter('inf' if x is None else x for x in bounds)
            of_error = CustomOverflowError("Invalid value at index {i}, not in ({bounds})", cls, bounds=bounds_str)

            low_bound, up_bound = bounds

            for i, value in enumerate(values):
                if low_bound is not None and value < low_bound:
                    if strict:
                        raise of_error(i=i)

                    values[i] = low_bound

                if up_bound is not None and value > up_bound:
                    if strict:
                        raise of_error(i=i)

                    values[i] = up_bound

            return values

        @overload
        @classmethod
        def from_param(cls, location: SLocationT | Literal[False]) -> Self:
            ...

        @overload
        @classmethod
        def from_param(cls, location: SLocationT | Literal[False] | None) -> Self | None:
            ...

        @classmethod
        def from_param(cls, location: SLocationT | Literal[False] | None) -> Self | None:
            if isinstance(location, SupportsFloatOrIndex) and location is not False:
                location = float(location)
                location = {0: location, 1: location}

            if location is None:
                return None

            if location is False:
                location = DFTTest.SLocation.NoProcess

            if isinstance(location, DFTTest.SLocation):
                return cls(list(location))

            return cls(location)

        def interpolate(
            self, method: InterMode = InterMode.LINEAR, res: int = 20, digits: int = 3
        ) -> DFTTest.SLocation:
            from scipy.interpolate import interp1d

            frequencies = list({round(x / (res - 1), digits) for x in range(res)})
            sigmas = interp1d(  # FIXME: interp1d is deprecated
                list(self.frequencies), list(self.sigmas), method.value, fill_value='extrapolate'
            )(frequencies)

            return DFTTest.SLocation(
                dict(zip(frequencies, [float(s) for s in sigmas])) | dict(zip(self.frequencies, self.sigmas)),
                strict=False
            )

        @classproperty
        def NoProcess(self) -> Self:
            return self.__class__({0: 0, 1: 0})

        class MultiDim:
            def __init__(
                self,
                horizontal: SLocationT | Literal[False] | None = None,
                vertical: SLocationT | Literal[False] | None = None,
                temporal: SLocationT | Literal[False] | None = None
            ) -> None:
                if not (horizontal or vertical or temporal):
                    raise CustomValueError('You must specify at least one dimension!', self.__class__)

                self.horizontal = DFTTest.SLocation.from_param(horizontal)
                self.vertical = DFTTest.SLocation.from_param(vertical)
                self.temporal = DFTTest.SLocation.from_param(temporal)

    class FilterType(CustomIntEnum):
        """Filtering types for DFTTest."""

        WIENER = 0
        """mult = max((psd - sigma) / psd, 0) ^ f0beta"""

        THR = 1
        """mult = psd < sigma ? 0.0 : 1.0"""

        MULT = 2
        """mult = sigma"""

        MULT_PSD = 3
        """mult = (psd >= pmin && psd <= pmax) ? sigma : sigma2"""

        MULT_RANGE = 4
        """mult = sigma * sqrt((psd * pmax) / ((psd + pmin) * (psd + pmax)))"""

    class SynthesisType(CustomIntEnum):
        """Synthesis type for spatial processing."""

        HANNING = 0
        HAMMING = 1
        NUTTALL = 10
        BLACKMAN = 2
        BLACKMAN_NUTTALL = 11
        BLACKMAN_HARRIS_4TERM = 3
        BLACKMAN_HARRIS_7TERM = 5
        KAISER_BESSEL = 4
        FLAT_TOP = 6
        RECTANGULAR = 7
        BARLETT = 8
        BARLETT_HANN = 9

    class _BackendBase(CustomEnum):
        kwargs: dict[str, Any]

        def DFTTest(self, clip: vs.VideoNode, *args: Any, **kwargs: Any) -> ConstantFormatVideoNode:
            self = self.resolve()

            if self == DFTTest.Backend.OLD:
                return core.dfttest.DFTTest(clip, *args, **self.kwargs | kwargs)

            try: 
                import dfttest2
            except ModuleNotFoundError as e:
                raise CustomRuntimeError("`dfttest2` python package is missing.", self.DFTTest) from e

            kwargs.update(backend=getattr(dfttest2.Backend, self.name)(**self.kwargs))

            return dfttest2.DFTTest(clip, *args, **kwargs)

        @cache
        def resolve(self) -> Self:
            if self.value != "auto":
                return self

            for member in list(self.__class__.__members__.values())[1:]:
                if hasattr(core, member.value):
                    return self.__class__(member.value)

            raise CustomRuntimeError(
                "No compatible plugin found. Please install one from: "

            )

        @property
        def plugin(self) -> Plugin:
            return getattr(core.lazy, self.resolve().value)

    class Backend(_BackendBase):
        def __init__(self, value: object, **kwargs: Any) -> None:
            self._value_ = value
            self.kwargs = kwargs

        AUTO = "auto"

        NVRTC = "dfttest2_nvrtc"
        HIPRTC = "dfttest2_hiprtc"
        cuFFT = "dfttest2_cuda"
        hipFFT = "dfttest2_hip"
        CPU = "dfttest2_cpu"
        GCC = "dfttest2_gcc"

        OLD = "dfttest"

        @overload
        def __call__(  # type: ignore [misc]
            self: Literal[DFTTest.Backend.NVRTC], *, device_id: int = 0, num_streams: int = 1
        ) -> DFTTest.Backend:
            ...

        @overload
        def __call__(  # type: ignore [misc]
            self: Literal[DFTTest.Backend.HIPRTC], *, device_id: int = 0, num_streams: int = 1
        ) -> DFTTest.Backend:
            ...

        @overload
        def __call__(  # type: ignore [misc]
            self: Literal[DFTTest.Backend.cuFFT], *, device_id: int = 0, in_place: bool = True
        ) -> DFTTest.Backend:
            ...

        @overload
        def __call__(  # type: ignore [misc]
            self: Literal[DFTTest.Backend.hipFFT], *, device_id: int = 0, in_place: bool = True
        ) -> DFTTest.Backend:
            ...

        @overload
        def __call__(self: Literal[DFTTest.Backend.CPU], *, opt: int = 0) -> DFTTest.Backend:  # type: ignore [misc]
            ...

        @overload
        def __call__(self: Literal[DFTTest.Backend.GCC]) -> DFTTest.Backend:  # type: ignore [misc]
            ...

        @overload
        def __call__(self: Literal[DFTTest.Backend.OLD], *, opt: int = ...) -> DFTTest.Backend:  # type: ignore [misc]
            ...

        def __call__(self, **kwargs: Any) -> DFTTest.Backend:
            new_enum = DFTTest._BackendBase(self.__class__.__name__, DFTTest.Backend.__members__)  # type: ignore
            member = getattr(new_enum, self.name)
            member.kwargs = kwargs
            return member

        if TYPE_CHECKING:
            def DFTTest(self, clip: vs.VideoNode, *args: Any, **kwargs: Any) -> ConstantFormatVideoNode:
                """_summary_

                :param clip: _description_
                :raises CustomRuntimeError: _description_
                :return: _description_
                """
                ...

            @cache
            def resolve(self) -> Self:
                """_summary_

                :raises CustomRuntimeError: _description_
                :return: _description_
                """
                ...

            @property
            def plugin(self) -> Plugin:
                """_summary_

                :return: _description_
                """
                ...

    def __init__(
        self,
        clip: vs.VideoNode | None = None,
        backend: Backend = Backend.AUTO,
        sloc: SLocationT | SLocation.MultiDim | None = None,
        **kwargs: Any
    ) -> None:
        self.clip = clip

        self.backend = backend

        self.default_slocation = sloc
        self.default_args = kwargs

    @overload
    def denoise(
        self,
        clip: vs.VideoNode,
        sloc: SLocationT | SLocation.MultiDim | None = None,
        /,
        tr: int = 0,
        ftype: int = FilterType.WIENER,
        swin: int | SynthesisType | None = None,
        twin: int | SynthesisType | None = None,
        planes: PlanesT = None,
        func: FuncExceptT | None = None,
        **kwargs: Any
    ) -> vs.VideoNode:
        ...

    @overload
    def denoise(
        self,
        sloc: SLocationT | SLocation.MultiDim,
        /,
        *,
        tr: int = 0,
        ftype: int = FilterType.WIENER,
        swin: int | SynthesisType | None = None,
        twin: int | SynthesisType | None = None,
        planes: PlanesT = None,
        func: FuncExceptT | None = None,
        **kwargs: Any
    ) -> vs.VideoNode:
        ...

    def denoise(
        self,
        clip_or_sloc: vs.VideoNode | SLocationT | SLocation.MultiDim,
        sloc: SLocationT | SLocation.MultiDim | None = None,
        /,
        tr: int = 0,
        ftype: int = FilterType.WIENER,
        swin: int | SynthesisType | None = None,
        twin: int | SynthesisType | None = None,
        planes: PlanesT = None,
        func: FuncExceptT | None = None,
        **kwargs: Any
    ) -> vs.VideoNode:
        func = func or self.denoise

        nclip: vs.VideoNode | None

        if isinstance(clip_or_sloc, vs.VideoNode):
            nclip = clip_or_sloc
            nsloc = fallback(sloc, self.default_slocation)
        else:
            nclip = self.clip
            nsloc = clip_or_sloc

        if nclip is None:
            raise CustomValueError('You must pass a clip!', func)

        assert check_progressive(nclip, func)

        ckwargs = dict[str, Any](
            tbsize=tr * 2 + 1,
            ftype=ftype,
            swin=swin,
            twin=twin,
            planes=planes
        )

        if isinstance(nsloc, DFTTest.SLocation.MultiDim):
            ckwargs.update(ssx=nsloc.horizontal, ssy=nsloc.vertical, sst=nsloc.temporal)
        else:
            ckwargs.update(slocation=DFTTest.SLocation.from_param(nsloc))  # type: ignore[call-overload]

        for k, v in ckwargs.items():
            if isinstance(v, DFTTest.SLocation):
                ckwargs[k] = list(flatten(v))

        return self.backend.DFTTest(nclip, **KwargsNotNone(ckwargs) | self.default_args | kwargs)

    @inject_self
    def extract_freq(
        self, clip: vs.VideoNode, sloc: SLocationT | SLocation.MultiDim, **kwargs: Any
    ) -> vs.VideoNode:
        kwargs = dict(func=self.extract_freq) | kwargs
        return clip.std.MakeDiff(self.denoise(clip, sloc, **kwargs))

    @inject_self
    def insert_freq(
        self, low: vs.VideoNode, high: vs.VideoNode, sloc: SLocationT | SLocation.MultiDim, **kwargs: Any
    ) -> vs.VideoNode:
        return low.std.MergeDiff(self.extract_freq(high, sloc, **dict(func=self.insert_freq) | kwargs))

    @inject_self
    def merge_freq(
        self, low: vs.VideoNode, high: vs.VideoNode, sloc: SLocationT | SLocation.MultiDim, **kwargs: Any
    ) -> vs.VideoNode:
        return self.insert_freq(
            self.denoise(low, sloc, **kwargs), high, sloc, **dict(func=self.merge_freq) | kwargs
        )


Frequency: TypeAlias = float
Sigma: TypeAlias = float


SLocationT = Union[
    float,
    DFTTest.SLocation,
    Sequence[Frequency | Sigma],
    Sequence[tuple[Frequency, Sigma]],
    Mapping[Frequency, Sigma]
]


@deprecated("`fft3d` is permanently deprecated and known to contain many bugs. Use with caution.")
def fft3d(clip: vs.VideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
    """
    Applies FFT3DFilter, a 3D frequency-domain filter used for strong denoising and mild sharpening.

    This filter processes frames using the Fast Fourier Transform (FFT) in the frequency domain.
    Unlike local filters, FFT3DFilter performs block-based, non-local processing.

    Official documentation:
    https://github.com/myrsloik/VapourSynth-FFT3DFilter/blob/master/doc/fft3dfilter.md

    Possibly faster implementation:
    https://github.com/AmusementClub/VapourSynth-FFT3DFilter/releases

    Note: Sigma values are internally scaled according to bit depth, unlike when using the plugin directly.

    :param clip:        Input video clip.
    :param **kwargs:    Additional parameters passed to the FFT3DFilter plugin.
    :return:            A heavily degraded version of DFTTest, with added banding and color shifts.
    """
    kwargs |= dict(interlaced=FieldBased.from_video(clip, False, fft3d).is_inter)

    # fft3dfilter requires sigma values to be scaled to bit depth
    # https://github.com/myrsloik/VapourSynth-FFT3DFilter/blob/master/doc/fft3dfilter.md#scaling-parameters-according-to-bit-depth
    sigma_multiplier = 1.0 / 256.0 if get_sample_type(clip) is vs.FLOAT else 1 << (get_depth(clip) - 8)

    for sigma in ['sigma', 'sigma2', 'sigma3', 'sigma4', 'smin ', 'smax']:
        if sigma in kwargs:
            kwargs[sigma] *= sigma_multiplier

    return core.fft3dfilter.FFT3DFilter(clip, **kwargs)
