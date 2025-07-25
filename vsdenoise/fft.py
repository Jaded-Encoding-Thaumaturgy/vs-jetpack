"""
This module implements wrappers for FFT (Fast Fourier Transform) based plugins.
"""

from __future__ import annotations

import math
from functools import cache
from typing import TYPE_CHECKING, Any, Iterator, Literal, Mapping, Sequence, TypeAlias, Union, overload

from jetpytools import KwargsNotNone, MismatchError, classproperty, fallback
from typing_extensions import Self, deprecated
from vapoursynth import Plugin

from vstools import (
    ConstantFormatVideoNode,
    CustomEnum,
    CustomIntEnum,
    CustomOverflowError,
    CustomRuntimeError,
    CustomValueError,
    FieldBased,
    FuncExceptT,
    PlanesT,
    SupportsFloatOrIndex,
    check_progressive,
    core,
    flatten,
    get_depth,
    get_sample_type,
    vs,
)

__all__ = ["DFTTest", "SLocationT", "fft3d"]


class _BackendBase(CustomEnum):
    kwargs: dict[str, Any]

    def DFTTest(self, clip: vs.VideoNode, *args: Any, **kwargs: Any) -> ConstantFormatVideoNode:  # noqa: N802
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
            "https://github.com/AmusementClub/vs-dfttest2 "
            "or https://github.com/HomeOfVapourSynthEvolution/VapourSynth-DFTTest "
        )

    @property
    def plugin(self) -> Plugin:
        return getattr(core.lazy, self.resolve().value)


Frequency: TypeAlias = float
"""
Represents the frequency value in the frequency domain.

The value is a `float` that represents the magnitude or position in the frequency spectrum.
"""
Sigma: TypeAlias = float
"""
Represents the sigma value, which is typically associated with noise standard deviation.

Used to indicate the level of noise or variance in the signal, and it is represented as a `float` value.
A higher sigma means more noise in the signal.
"""


class DFTTest:
    """
    2D/3D frequency domain denoiser using Discrete Fourier transform.
    """

    class SLocation(Mapping[Frequency, Sigma]):
        """
        A helper class for handling sigma values as functions of frequency in different dimensions.

        This class allows you to specify and interpolate sigma values at different frequency locations.
        The frequency range is normalized to [0.0,1.0] with 0.0 being the lowest frequency and 1.0 being the highest.
        The class also supports handling sigma values as a function of horizontal (ssx), vertical (ssy),
        and temporal (sst) frequencies.
        """

        class InterMode(CustomEnum):
            """
            Interpolation modes for sigma values in `SLocation`.

            Defines how sigma values should be interpolated across frequency locations.
            """

            LINEAR = "linear"
            """
            Linear Interpolation:

            Performs a linear interpolation between given frequency and sigma values.
            This is the default interpolation method.

            This method connects adjacent points with straight lines and is typically
            used when data points change at a constant rate.
            """

            SPLINE = 1
            """
            Spline Interpolation:

            Performs a spline interpolation between given frequency and sigma values.

            This method fits a smooth curve (using spline functions) through the given data points,
            resulting in a smoother transition between values than linear interpolation.
            """

            SPLINE_LINEAR = "slinear"
            """
            Spline-Linear Interpolation:

            A combination of spline interpolation with linear interpolation for smoother transitions.

            This method combines spline interpolation with linear methods,
            making it useful for smoother transitions but retaining linear simplicity where needed.
            """

            QUADRATIC = "quadratic"
            """
            Quadratic Interpolation:

            Performs quadratic interpolation, fitting a second-degree polynomial through the data points.

            This method creates a parabolic curve between points, which can provide better smoothness
            for certain types of data.
            """

            CUBIC = "cubic"
            """
            Cubic Interpolation:

            Performs cubic interpolation, fitting a third-degree polynomial through the data points.

            This is commonly used to produce smooth and visually appealing curves,
            especially when higher smoothness is needed between data points.
            """

            NEAREST = "nearest"
            """
            Nearest Neighbor Interpolation:

            Uses the nearest value for interpolation.
            This method does not smooth the data, but rather uses the value closest to the target frequency.

            This can result in discontinuous transitions, especially if the data is not uniform.
            """

            NEAREST_UP = "nearest-up"
            """
            Nearest Neighbor with Rounding Up:

            A variation of nearest neighbor interpolation where the value is always rounded up.

            This can be useful when you want to ensure that the sigma value is never lower than a certain threshold,
            avoiding underestimation.
            """

            ZERO = "zero"
            """
            Zero Order Hold (ZOH):

            A simple method that holds the last value before the target frequency location.

            This is equivalent to piecewise constant interpolation, where the sigma value is "held"
            at the last known value until the next point.
            """

            @overload
            def __call__(self, location: SLocationT, /, *, res: int = 20, digits: int = 3) -> DFTTest.SLocation:
                """
                Interpolates sigma values at a given frequency location.

                Args:
                    location: The frequency location at which to interpolate sigma.
                    res: The resolution of the interpolation (default is 20).
                    digits: The precision of the frequency values (default is 3 decimal places).

                Returns:
                    The interpolated `SLocation` object.
                """

            @overload
            def __call__(
                self, location: SLocationT | None, /, *, res: int = 20, digits: int = 3
            ) -> DFTTest.SLocation | None:
                """
                Interpolates sigma values for a given location or returns `None` if no location is provided.

                Args:
                    location: The frequency location or `None` for no interpolation.
                    res: The resolution of the interpolation (default is 20).
                    digits: The precision of the frequency values (default is 3 decimal places).

                Returns:
                    The interpolated `SLocation` object or `None` if no location is provided.
                """

            @overload
            def __call__(
                self,
                h_loc: SLocationT | None,
                v_loc: SLocationT | None,
                t_loc: SLocationT | None,
                /,
                *,
                res: int = 20,
                digits: int = 3,
            ) -> DFTTest.SLocation.MultiDim:
                """
                Interpolates sigma values for horizontal, vertical, and temporal locations.

                Args:
                    h_loc: The horizontal frequency location.
                    v_loc: The vertical frequency location.
                    t_loc: The temporal frequency location.
                    res: The resolution of the interpolation (default is 20).
                    digits: The precision of the frequency values (default is 3 decimal places).

                Returns:
                    A `MultiDim` object containing the interpolated values.
                """

            def __call__(
                self, *locations: SLocationT | None, res: int = 20, digits: int = 3
            ) -> DFTTest.SLocation | None | DFTTest.SLocation.MultiDim:
                """
                Interpolates sigma values for given frequency locations. Can handle multiple locations for horizontal,
                vertical, and temporal frequencies.

                Args:
                    *locations: The frequency locations for interpolation.
                    res: The resolution of the interpolation (default is 20).
                    digits: The precision of the frequency values (default is 3 decimal places).

                Returns:
                    The interpolated `SLocation` object or a `MultiDim` object if multiple locations are provided.
                """
                if len(locations) == 1:
                    sloc = DFTTest.SLocation.from_param(locations[0])

                    if sloc is not None:
                        sloc = sloc.interpolate(self, res, digits)

                    return sloc

                return DFTTest.SLocation.MultiDim(*(self(x, res=res, digits=digits) for x in locations))

        frequencies: tuple[Frequency, ...]
        """
        The list of frequency locations.
        """

        sigmas: tuple[Sigma, ...]
        """
        The corresponding sigma values for each frequency.
        """

        def __init__(
            self,
            locations: Sequence[Frequency | Sigma] | Sequence[tuple[Frequency, Sigma]] | Mapping[Frequency, Sigma],
            interpolate: InterMode | None = None,
            strict: bool = True,
        ) -> None:
            """
            Initializes the SLocation object by processing frequency-sigma pairs and sorting them.

            Example:
                ```py
                sloc = DFTTest.SLocation(
                    [(0.0, 4), (0.25, 8), (0.5, 10), (0.75, 32), (1.0, 64)], DFTTest.SLocation.InterMode.SPLINE
                )
                >>> sloc  # rounded to 3 digits
                >>> {
                    0.0: 4, 0.053: 4.848, 0.105: 5.68, 0.158: 6.528, 0.211: 7.376, 0.25: 8, 0.263: 8.104, 0.316: 8.528,
                    0.368: 8.944, 0.421: 9.368, 0.474: 9.792, 0.5: 10, 0.526: 12.288, 0.579: 16.952, 0.632: 21.616,
                    0.684: 26.192, 0.737: 30.856, 0.75: 32, 0.789: 36.992, 0.842: 43.776, 0.895: 50.56, 0.947: 57.216,
                    1.0: 64
                }
                ```

            Args:
                locations: A sequence of tuples or a dictionary that specifies frequency and sigma pairs.
                interpolate: The interpolation method to be used for sigma values. If `None`, no interpolation is done.
                strict: If `True`, raises an error if values are out of bounds.
                    If `False`, it will clamp values to the bounds.

            Raises:
                CustomValueError: If `locations` has not an even number of items.
            """
            if isinstance(locations, Mapping):
                frequencies, sigmas = list(locations.keys()), list(locations.values())
            else:
                locations = list[float](flatten(locations))

                if len(locations) % 2:
                    raise CustomValueError(
                        "locations must resolve to an even number of items, pairing frequency and sigma respectively",
                        self.__class__,
                    )

                frequencies, sigmas = list(locations[0::2]), list(locations[1::2])

            frequencies = self.bounds_check(frequencies, (0, 1), strict)
            sigmas = self.bounds_check(sigmas, (0, math.inf), strict)

            self.frequencies, self.sigmas = (t for t in zip(*sorted(zip(frequencies, sigmas))))

            if interpolate:
                interpolated = self.interpolate(interpolate)

                self.frequencies, self.sigmas = interpolated.frequencies, interpolated.sigmas

        def __str__(self) -> str:
            return str(dict(self))

        def __getitem__(self, key: Frequency) -> Sigma:
            """
            Get the sigma value associated with a given frequency.

            Args:
                key: The frequency value to look up.

            Returns:
                The sigma value corresponding to the given frequency.

            Raises:
                KeyError: If the frequency is not found.
            """
            try:
                return self.sigmas[self.frequencies.index(key)]
            except ValueError as e:
                raise KeyError(key) from e

        def __iter__(self) -> Iterator[float]:
            """
            Return an iterator over the frequencies.

            Returns:
                An iterator yielding each frequency in order.
            """
            yield from self.frequencies

        def __len__(self) -> int:
            """
            Return the number of frequency-sigma pairs.

            Returns:
                The number of pairs.

            Raises:
                MismatchError: If the number of frequencies and sigmas does not match.
            """
            if len(self.frequencies) == (length := len(self.sigmas)):
                return length

            raise MismatchError(
                self.__class__,
                [self.frequencies, self.sigmas],
                "Frequencies and sigmas must have the same number of items.",
            )

        def __reversed__(self) -> Self:
            """
            Reverses the frequency-sigma pairs, inverting the frequency range and reversing the order of sigma values.

            Returns:
                A new `SLocation` instance with reversed frequency-sigma pairs.
            """
            return self.__class__(dict(zip((1 - f for f in reversed(self.frequencies)), list(reversed(self.sigmas)))))

        @classmethod
        def bounds_check(
            cls, values: Sequence[float], bounds: tuple[float | None, float | None], strict: bool = False
        ) -> list[float]:
            """
            Checks and bounds the sigma values to the specified limits.

            Args:
                values: The list of sigma values to check.
                bounds: The valid bounds for the sigma values.
                strict: If `True`, raises an error for out-of-bounds values, otherwise clamps them to bounds.

            Returns:
                A list of sigma values that are within the specified bounds.
            """
            if not values:
                raise CustomValueError('"values" can\'t be empty!', cls)

            values = list(values)

            of_error = CustomOverflowError(
                "Invalid value at index {i}, not in ({bounds})",
                cls,
                bounds=[math.inf if x is None else x for x in bounds],
            )

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
        def from_param(cls, location: SLocationT | Literal[False]) -> Self: ...

        @overload
        @classmethod
        def from_param(cls, location: SLocationT | Literal[False] | None) -> Self | None: ...

        @classmethod
        def from_param(cls, location: SLocationT | Literal[False] | None) -> Self | None:
            """
            Converts a frequency-sigma pair or a literal `False` to an `SLocation` instance.
            Returns `None` if no processing.

            Args:
                location: A frequency-sigma pair, `False` for no processing, or `None`.

            Returns:
                An `SLocation` instance or `None`.
            """
            if isinstance(location, SupportsFloatOrIndex) and location is not False:
                location = float(location)
                location = {0: location, 1: location}

            if location is None:
                return None

            if location is False:
                return cls.NoProcess

            return cls(location)

        def interpolate(
            self, method: InterMode = InterMode.LINEAR, res: int = 20, digits: int = 3
        ) -> DFTTest.SLocation:
            """
            Interpolates the sigma values across a specified resolution.

            Args:
                method: The interpolation method to use.
                res: The resolution of the interpolation (default is 20).
                digits: The precision of the frequency values (default is 3 decimal places).

            Returns:
                A new `SLocation` instance with interpolated sigma values.
            """
            from scipy.interpolate import interp1d

            frequencies = list({round(x / (res - 1), digits) for x in range(res)})
            sigmas = interp1d(  # FIXME: interp1d is deprecated
                list(self.frequencies), list(self.sigmas), method.value, fill_value="extrapolate"
            )(frequencies)

            return self.__class__(dict(zip(frequencies, (float(s) for s in sigmas))) | dict(self), strict=False)

        @classproperty
        @classmethod
        def NoProcess(cls) -> Self:  # noqa: N802
            """
            Returns a pre-defined `SLocation` instance that performs no processing
            (i.e., sigma is zero for all locations).

            Returns:
                A `SLocation` instance with no processing.
            """
            return cls({0: 0, 1: 0})

        class MultiDim:
            """
            A helper class for handling multi-dimensional frequency-sigma mappings for horizontal, vertical,
            and temporal dimensions.
            """

            def __init__(
                self,
                horizontal: SLocationT | Literal[False] | None = None,
                vertical: SLocationT | Literal[False] | None = None,
                temporal: SLocationT | Literal[False] | None = None,
            ) -> None:
                """
                Initializes a `MultiDim` object with specified frequency-sigma mappings for horizontal,
                vertical, and temporal dimensions.

                Example:
                    Denoise only on the vertical dimension:
                    ```py
                    sloc = DFTTest.SLocation.MultiDim(
                        vertical=[(0.0, 8.0), (0.25, 16.0), (0.5, 0.0), (0.75, 16.0), (1.0, 0.0)]
                    )

                    denoised = DFTTest(clip).denoise(sloc)
                    ```

                Args:
                    horizontal: The sigma values for horizontal frequency locations.
                    vertical: The sigma values for vertical frequency locations.
                    temporal: The sigma values for temporal frequency locations.

                Raises:
                    CustomValueError: If no dimension is specified.
                """
                if not (horizontal or vertical or temporal):
                    raise CustomValueError("You must specify at least one dimension!", self.__class__)

                self.horizontal = DFTTest.SLocation.from_param(horizontal)
                self.vertical = DFTTest.SLocation.from_param(vertical)
                self.temporal = DFTTest.SLocation.from_param(temporal)

    class FilterType(CustomIntEnum):
        """
        Enumeration of filtering types used in DFTTest plugin.

        These filters define how the real and imaginary parts of each complex DFT coefficient
        are scaled (via `mult`) during denoising, based on their power spectrum density (PSD).

        The term *psd* refers to the [power spectral density](https://en.wikipedia.org/wiki/Spectral_power_distribution),
        computed as `psd = real² + imag²`.
        """

        WIENER = 0
        """
        Generalized Wiener filter.

        Suppresses noise while preserving signal using an adaptive formula.

        Formula:
        ```
        mult = max((psd - sigma) / psd, 0) ** f0beta
        ```
        """

        THR = 1
        """
        Hard threshold filter.

        Removes frequency components below a given PSD threshold.

        Formula:
        ```
        mult = 0.0 if psd < sigma else 1.0
        ```
        """

        MULT = 2
        """
        Constant multiplier.

        Applies a fixed scaling factor to all coefficients, regardless of PSD.

        Formula:
        ```
        mult = sigma
        ```
        """

        MULT_PSD = 3
        """
        Conditional multiplier based on PSD range.

        Switches between two scaling values depending on whether `psd` falls within a specific range.

        Formula:
        ```
        mult = sigma  if (pmin <= psd <= pmax)
            sigma2 otherwise
        ```
        """

        MULT_RANGE = 4
        """
        Smooth range-based multiplier.

        Computes a smooth gain based on PSD and specified min/max bounds.

        Formula:
        ```
        mult = sigma * sqrt((psd * pmax) / ((psd + pmin) * (psd + pmax)))
        ```
        """

    class SynthesisType(CustomIntEnum):
        """
        Enumeration of synthesis window types used in DFTTest plugin.

        These constants are used with the `swin` (spatial) and `twin` (temporal) parameters
        to specify the window function applied during frequency-domain processing.
        """

        HANNING = 0
        """
        Hanning window (0). A raised cosine window with good frequency resolution and low spectral leakage.

        See: https://en.wikipedia.org/wiki/Window_function#Hann_(Hanning)_window
        """

        HAMMING = 1
        """
        Hamming window (1). Similar to Hanning but with slightly reduced side lobe attenuation.

        See: https://en.wikipedia.org/wiki/Window_function#Hamming_window
        """

        BLACKMAN = 2
        """
        Blackman window (2). Better side lobe attenuation than Hamming or Hanning at the cost of wider main lobe.

        See: https://en.wikipedia.org/wiki/Window_function#Blackman_window
        """

        BLACKMAN_HARRIS_4TERM = 3
        """
        4-term Blackman-Harris window (3). Low side lobes, ideal for high dynamic range applications.

        See: https://en.wikipedia.org/wiki/Window_function#Blackman%E2%80%93Harris_window
        """

        KAISER_BESSEL = 4
        """
        Kaiser-Bessel window (4). Adjustable window (beta parameter) balancing resolution and leakage.

        See: https://en.wikipedia.org/wiki/Window_function#Kaiser_window
        """

        BLACKMAN_HARRIS_7TERM = 5
        """
        7-term Blackman-Harris window (5). Extended version with even greater side lobe suppression.

        See:
            - https://ccrma.stanford.edu/~jos/Windows/Blackman_Harris_Window_Family.html
            - https://github.com/hukenovs/blackman_harris_win
        """

        FLAT_TOP = 6
        """
        Flat top window (6). Optimized for amplitude accuracy and minimizing scalloping loss.

        See: https://en.wikipedia.org/wiki/Window_function#Flat_top_window
        """

        RECTANGULAR = 7
        """
        Rectangular window (7). Equivalent to no windowing; highest resolution, but worst leakage.

        See: https://en.wikipedia.org/wiki/Window_function#Rectangular_window
        """

        BARLETT = 8
        """
        Bartlett window (8). Also known as a triangular window; moderate leakage reduction.

        See: https://en.wikipedia.org/wiki/Window_function#Bartlett_window
        """

        BARLETT_HANN = 9
        """
        Bartlett-Hann window (9). Hybrid of Bartlett and Hanning with a smoother taper.

        See: https://en.wikipedia.org/wiki/Window_function#Bartlett%E2%80%93Hann_window
        """

        NUTTALL = 10
        """
        Nuttall window (10). Four-term cosine window with very low side lobes.

        See: https://en.wikipedia.org/wiki/Window_function#Nuttall_window,_continuous_first_derivative
        """

        BLACKMAN_NUTTALL = 11
        """
        Blackman-Nuttall window (11). Variant of Nuttall with even stronger side lobe suppression.

        See: https://en.wikipedia.org/wiki/Window_function#Blackman%E2%80%93Nuttall_window
        """

    class Backend(_BackendBase):
        """
        Enum representing available backends on which to run the plugin.
        """

        def __init__(self, value: object, **kwargs: Any) -> None:
            self._value_ = value
            self.kwargs = kwargs

        AUTO = "auto"
        """
        Automatically select the most appropriate backend based on system capabilities.
        Typically prioritizes GPU backends if available, otherwise falls back to CPU.
        """

        NVRTC = "dfttest2_nvrtc"
        """
        NVIDIA GPU backend using NVRTC (NVIDIA Runtime Compilation).
        """

        HIPRTC = "dfttest2_hiprtc"
        """
        AMD GPU backend using HIPRTC (HIP Runtime Compilation).
        """

        cuFFT = "dfttest2_cuda"  # noqa: N815
        """
        NVIDIA GPU backend using precompiled CUDA and cuFFT.
        """

        hipFFT = "dfttest2_hip"  # noqa: N815
        """
        AMD GPU backend using precompiled HIP and hipFFT.
        """

        CPU = "dfttest2_cpu"
        """
        Modern CPU backend using optimized multi-threaded CPU code.
        """

        GCC = "dfttest2_gcc"
        """
        CPU backend compiled with GCC.
        """

        OLD = "dfttest"
        """
        Legacy DFTTest implementation by HolyWu.
        """

        @overload
        def __call__(  # type: ignore [misc]
            self: Literal[DFTTest.Backend.NVRTC], *, device_id: int = 0, num_streams: int = 1
        ) -> DFTTest.Backend:
            """
            Configures the NVRTC (NVIDIA Runtime Compilation) backend for DFTTest.

            Args:
                device_id: The index of the GPU device to use (default is 0, the first GPU).
                num_streams: The number of CUDA streams to use for parallel computation (default is 1).

            Returns:
                The configured backend.
            """

        @overload
        def __call__(  # type: ignore [misc]
            self: Literal[DFTTest.Backend.HIPRTC], *, device_id: int = 0, num_streams: int = 1
        ) -> DFTTest.Backend:
            """
            Configures the HIPRTC (HIP Runtime Compilation) backend for DFTTest.

            Args:
                device_id: The index of the AMD GPU device to use (default is 0).
                num_streams: The number of HIP streams to use for computation (default is 1).

            Returns:
                The configured backend.
            """

        @overload
        def __call__(  # type: ignore [misc]
            self: Literal[DFTTest.Backend.cuFFT], *, device_id: int = 0, in_place: bool = True
        ) -> DFTTest.Backend:
            """
            Configures the cuFFT (CUDA Fast Fourier Transform) backend for DFTTest.

            Args:
                device_id: The index of the CUDA-enabled GPU device to use (default is 0).
                in_place: If True, computes the FFT in-place, modifying the input array (default is True).

            Returns:
                The configured backend.
            """

        @overload
        def __call__(  # type: ignore [misc]
            self: Literal[DFTTest.Backend.hipFFT], *, device_id: int = 0, in_place: bool = True
        ) -> DFTTest.Backend:
            """
            Configures the hipFFT (HIP Fast Fourier Transform) backend for DFTTest.

            Args:
                device_id: The index of the AMD GPU device to use (default is 0).
                in_place: If True, computes the FFT in-place, modifying the input array (default is True).

            Returns:
                The configured backend.
            """

        @overload
        def __call__(self: Literal[DFTTest.Backend.CPU], *, opt: int = ...) -> DFTTest.Backend:  # type: ignore [misc]
            """
            Configures the CPU backend for DFTTest.

            Args:
                opt: CPU optimization level (default is auto detect).

            Returns:
                The configured backend.
            """

        @overload
        def __call__(self: Literal[DFTTest.Backend.GCC]) -> DFTTest.Backend:  # type: ignore [misc]
            """
            Configures the GCC backend for DFTTest.

            Returns:
                The configured backend.
            """

        @overload
        def __call__(self: Literal[DFTTest.Backend.OLD], *, opt: int = ...) -> DFTTest.Backend:  # type: ignore [misc]
            """
            Configures the legacy DFTTest (OLD) backend.

            Args:
                opt: CPU optimization level (default is auto detect).

            Returns:
                The configured backend.
            """

        def __call__(self, **kwargs: Any) -> DFTTest.Backend:
            """
            This method is used to apply the specified backend configuration with provided keyword arguments.

            Depending on the backend, the arguments may represent device IDs, streams,
            or other backend-specific settings.

            Args:
                **kwargs: Additional configuration parameters for the backend.

            Returns:
                The configured backend with applied parameters.
            """
            new_enum = _BackendBase(self.__class__.__name__, DFTTest.Backend.__members__)  # type: ignore
            member = getattr(new_enum, self.name)
            member.kwargs = kwargs
            return member

        if TYPE_CHECKING:
            kwargs: dict[str, Any]
            """
            Additional configuration parameters for the backend.
            """

            def DFTTest(self, clip: vs.VideoNode, *args: Any, **kwargs: Any) -> ConstantFormatVideoNode:  # noqa: N802
                """
                Applies the DFTTest denoising filter using the plugin associated with the selected backend.

                Args:
                    clip: Source clip.
                    *args: Positional arguments passed to the selected plugin.
                    **kwargs: Keyword arguments passed to the selected plugin.

                Raises:
                    CustomRuntimeError: If the selected backend is not available or unsupported.

                Returns:
                    Denoised clip.
                """
                ...

            @cache
            def resolve(self) -> Self:
                """
                Resolves the appropriate DFTTest backend to use based on availability.

                If the current instance is not DFTTest.Backend.AUTO, it returns itself.
                Otherwise, it attempts to select the best available backend.

                Raises:
                    CustomRuntimeError: If no supported DFTTest implementation is available on the system.

                Returns:
                    The resolved DFTTest.Backend to use for processing.
                """
                ...

            @property
            def plugin(self) -> Plugin:
                """
                Returns the appropriate DFTTest plugin based on the current backend.

                Returns:
                    The corresponding DFTTest plugin for the resolved backend.
                """
                ...

    def __init__(
        self,
        clip: vs.VideoNode | None = None,
        backend: Backend = Backend.AUTO,
        sloc: SLocationT | SLocation.MultiDim | None = None,
        tr: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the `DFTTest` class with the provided clip, backend, and frequency location.

        Example:
            ```py
            dfttest = DFTTest(clip, DFTTest.Backend.OLD)
            denoised_low_frequencies = dfttest.denoise({0: 16, 0.25: 8, 0.5: 0, 1.0: 0})
            denoised_high_frequencies = dfttest.denoise([(0, 0), (0.5, 0), (0.75, 16), (1.0, 32)])
            ```

        Args:
            clip: Source clip.
            backend: The backend to use processing.
            sloc: The frequency location for denoising.
            tr: Temporal radius for denoising (`tr` * 2 + 1 == `tbsize`).
                Note: Unlike the default plugin implementation, the default value here is tr=0 (i.e. `tbsize=1`).
            **kwargs: Additional parameters to configure the denoising process.
        """
        self.clip = clip

        self.backend = backend

        self.default_slocation = sloc
        self.default_tr = tr
        self.default_args = kwargs

    @overload
    def denoise(
        self,
        clip: vs.VideoNode,
        sloc: SLocationT | SLocation.MultiDim | None = None,
        /,
        tr: int | None = None,
        ftype: int = FilterType.WIENER,
        swin: int | SynthesisType | None = None,
        twin: int | SynthesisType | None = None,
        planes: PlanesT = None,
        func: FuncExceptT | None = None,
        **kwargs: Any,
    ) -> vs.VideoNode: ...

    @overload
    def denoise(
        self,
        sloc: SLocationT | SLocation.MultiDim,
        /,
        *,
        tr: int | None = None,
        ftype: int = FilterType.WIENER,
        swin: int | SynthesisType | None = None,
        twin: int | SynthesisType | None = None,
        planes: PlanesT = None,
        func: FuncExceptT | None = None,
        **kwargs: Any,
    ) -> vs.VideoNode: ...

    def denoise(
        self,
        clip_or_sloc: vs.VideoNode | SLocationT | SLocation.MultiDim,
        sloc: SLocationT | SLocation.MultiDim | None = None,
        /,
        tr: int | None = None,
        ftype: int = FilterType.WIENER,
        swin: int | SynthesisType | None = None,
        twin: int | SynthesisType | None = None,
        planes: PlanesT = None,
        func: FuncExceptT | None = None,
        **kwargs: Any,
    ) -> vs.VideoNode:
        """
        Denoises a clip using Discrete Fourier Transform (DFT).

        More informations:
            - [VapourSynth DFTTest plugin](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-DFTTest/blob/master/README.md)
            - [AviSynth DFTTest docs](http://avisynth.nl/index.php/Dfttest)
            - [vs-dfttest2 docstring](https://github.com/AmusementClub/vs-dfttest2/blob/573bb36c53df93c46a38926c7c654569e3679732/dfttest2.py#L614-L764)

        Examples:
            Apply a constant sigma:
            ```py
            denoised = DFTTest().denoise(clip, sigma=16)
            ```

            Use frequency-dependent sigma values:
            ```py
            denoised = DFTTest().denoise(clip, {0: 0, 0.25: 4, 0.5: 8, 0.75: 16, 1.0: 32})
            ```

        Args:
            clip_or_sloc: Either a video clip or frequency location.
            sloc: Frequency location (used if `clip_or_sloc` is a video clip).
            tr: Temporal radius for denoising (`tr` * 2 + 1 == `tbsize`).
                Note: Unlike the default plugin implementation, the default value here is tr=0 (i.e. `tbsize=1`).
            ftype: Filter type for denoising (see [FilterType][vsdenoise.fft.DFTTest.FilterType] enum).
            swin: Synthesis window size (can use [SynthesisType][vsdenoise.fft.DFTTest.SynthesisType] enum).
            twin: Temporal window size (can use [SynthesisType][vsdenoise.fft.DFTTest.SynthesisType] enum).
            planes: Planes to apply the denoising filter.
            func: Function returned for custom error handling.
            **kwargs: Additional parameters for the denoising process.

        Returns:
            The denoised video node.
        """
        func = func or self.denoise

        nclip: vs.VideoNode | None

        if isinstance(clip_or_sloc, vs.VideoNode):
            nclip = clip_or_sloc
            nsloc = self.default_slocation if sloc is None else sloc
        else:
            nclip = self.clip
            nsloc = clip_or_sloc

        if nclip is None:
            raise CustomValueError("You must pass a clip!", func)

        assert check_progressive(nclip, func)

        ckwargs = dict[str, Any](
            tbsize=fallback(tr, self.default_tr, 0) * 2 + 1, ftype=ftype, swin=swin, twin=twin, planes=planes
        )

        if isinstance(nsloc, DFTTest.SLocation.MultiDim):
            ckwargs.update(ssx=nsloc.horizontal, ssy=nsloc.vertical, sst=nsloc.temporal)
        else:
            ckwargs.update(slocation=DFTTest.SLocation.from_param(nsloc))

        for k, v in ckwargs.items():
            if isinstance(v, DFTTest.SLocation):
                ckwargs[k] = list[float](flatten(v.items()))

        return self.backend.DFTTest(nclip, **KwargsNotNone(ckwargs) | self.default_args | kwargs)

    def extract_freq(self, clip: vs.VideoNode, sloc: SLocationT | SLocation.MultiDim, **kwargs: Any) -> vs.VideoNode:
        """
        Extracts the frequency domain from the given clip by subtracting the denoised clip from the original.

        Args:
            clip: The clip from which the frequency domain is to be extracted.
            sloc: The frequency location for the extraction process.
            **kwargs: Additional parameters for the extraction process.

        Returns:
            The clip with the extracted frequency domain.
        """
        kwargs = {"func": self.extract_freq} | kwargs
        return clip.std.MakeDiff(self.denoise(clip, sloc, **kwargs))

    def insert_freq(
        self, low: vs.VideoNode, high: vs.VideoNode, sloc: SLocationT | SLocation.MultiDim, **kwargs: Any
    ) -> vs.VideoNode:
        """
        Inserts the frequency domain from one clip into another by merging the frequency information.

        Args:
            low: The low-frequency component clip.
            high: The high-frequency component clip.
            sloc: The frequency location for the merging process.
            **kwargs: Additional parameters for the merging process.

        Returns:
            The merged clip with the inserted frequency domain.
        """
        return low.std.MergeDiff(self.extract_freq(high, sloc, **{"func": self.insert_freq} | kwargs))

    def merge_freq(
        self, low: vs.VideoNode, high: vs.VideoNode, sloc: SLocationT | SLocation.MultiDim, **kwargs: Any
    ) -> vs.VideoNode:
        """
        Merges the low and high-frequency components by applying denoising to the low-frequency component.

        Args:
            low: The low-frequency component clip.
            high: The high-frequency component clip.
            sloc: The frequency location for the merging process.
            **kwargs: Additional parameters for the merging process.

        Returns:
            The merged clip with the denoised low-frequency and high-frequency components.
        """
        return self.insert_freq(self.denoise(low, sloc, **kwargs), high, sloc, **{"func": self.merge_freq} | kwargs)


SLocationT = Union[
    int,
    float,
    DFTTest.SLocation,
    Sequence[Frequency | Sigma],
    Sequence[tuple[Frequency, Sigma]],
    Mapping[Frequency, Sigma],
]
"""
A type that represents various ways to specify a location in the frequency domain for denoising operations.

The `SLocationT` type can be one of the following:

- `int` or `float`:
  A single frequency value (for 1D frequency location).
- `DFTTest.SLocation`:
  A structured class for defining frequency locations in a more complex manner.
- `Sequence[Frequency, Sigma]`:
  A sequence (e.g., list or tuple) that alternates between `Frequency` and `Sigma` values.
  The sequence must have an even number of items, where each frequency is followed by its corresponding sigma.
  For example: `[0.0, 8.0, 0.5, 16.0]` where `0.0` is the frequency and `8.0` is its corresponding sigma, and so on.
- `Sequence[tuple[Frequency, Sigma]]`:
  A sequence of tuples, where each tuple contains a `Frequency` and a `Sigma`.
- `Mapping[Frequency, Sigma]`:
  A dictionary (Mapping) where each `Frequency` key maps to a corresponding `Sigma` value.

The sequence or mapping must represent a pairing of frequency and sigma values for denoising operations.
In the case of a sequence like `Sequence[Frequency, Sigma]`, it is essential that the number of items is even,
ensuring every frequency has an associated sigma.
"""


@deprecated("`fft3d` is permanently deprecated and known to contain many bugs. Use with caution.")
def fft3d(clip: vs.VideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
    """
    Applies FFT3DFilter, a 3D frequency-domain filter used for strong denoising and mild sharpening.

    This filter processes frames using the Fast Fourier Transform (FFT) in the frequency domain.
    Unlike local filters, FFT3DFilter performs block-based, non-local processing.

       - [Official documentation](https://github.com/myrsloik/VapourSynth-FFT3DFilter/blob/master/doc/fft3dfilter.md)
       - [Possibly faster implementation](https://github.com/AmusementClub/VapourSynth-FFT3DFilter/releases)

    Note: Sigma values are internally scaled according to bit depth, unlike when using the plugin directly.

    Args:
        clip: Input video clip.
        **kwargs: Additional parameters passed to the FFT3DFilter plugin.

    Returns:
        A heavily degraded version of DFTTest, with added banding and color shifts.
    """
    kwargs |= {"interlaced": FieldBased.from_video(clip, False, fft3d).is_inter}

    # fft3dfilter requires sigma values to be scaled to bit depth
    # https://github.com/myrsloik/VapourSynth-FFT3DFilter/blob/master/doc/fft3dfilter.md#scaling-parameters-according-to-bit-depth
    sigma_multiplier = 1.0 / 256.0 if get_sample_type(clip) is vs.FLOAT else 1 << (get_depth(clip) - 8)

    for sigma in ["sigma", "sigma2", "sigma3", "sigma4", "smin ", "smax"]:
        if sigma in kwargs:
            kwargs[sigma] *= sigma_multiplier

    return core.fft3dfilter.FFT3DFilter(clip, **kwargs)
