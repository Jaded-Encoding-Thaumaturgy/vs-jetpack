import typing
import vapoursynth as vs
from dataclasses import dataclass

if typing.TYPE_CHECKING:
    from vstools import ConstantFormatVideoNode

__all__ = ["DFTTest", "DFTTest2", "Backend"]

__version__: str

class Backend:
    @dataclass(frozen=False)
    class cuFFT:
        device_id: int = ...
        in_place: bool = ...

    @dataclass(frozen=False)
    class NVRTC:
        device_id: int = ...
        num_streams: int = ...

    @dataclass(frozen=False)
    class CPU:
        opt: int = ...

    @dataclass(frozen=False)
    class GCC: ...

    @dataclass(frozen=False)
    class hipFFT:
        device_id: int = ...
        in_place: bool = ...

    @dataclass(frozen=False)
    class HIPRTC:
        device_id: int = ...
        num_streams: int = ...

backendT = Backend.cuFFT | Backend.NVRTC | Backend.CPU | Backend.GCC | Backend.hipFFT | Backend.HIPRTC

def DFTTest2(
    clip: vs.VideoNode,
    ftype: typing.Literal[0, 1, 2, 3, 4] = 0,
    sigma: float | typing.Sequence[typing.Callable[[float], float]] = 8.0,
    sigma2: float = 8.0,
    pmin: float = 0.0,
    pmax: float = 500.0,
    sbsize: int = 16,
    sosize: int = 12,
    tbsize: int = 3,
    swin: typing.Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] = 0,
    twin: typing.Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] = 7,
    sbeta: float = 2.5,
    tbeta: float = 2.5,
    zmean: bool = True,
    f0beta: float = 1.0,
    ssystem: typing.Literal[0, 1] = 0,
    planes: int | typing.Sequence[int] | None = None,
    backend: backendT = ...,
) -> ConstantFormatVideoNode: ...

FREQ = float
SIGMA = float

def DFTTest(
    clip: vs.VideoNode,
    ftype: typing.Literal[0, 1, 2, 3, 4] = 0,
    sigma: float = 8.0,
    sigma2: float = 8.0,
    pmin: float = 0.0,
    pmax: float = 500.0,
    sbsize: int = 16,
    smode: typing.Literal[0, 1] = 1,
    sosize: int = 12,
    tbsize: int = 3,
    swin: typing.Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] = 0,
    twin: typing.Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] = 7,
    sbeta: float = 2.5,
    tbeta: float = 2.5,
    zmean: bool = True,
    f0beta: float = 1.0,
    nlocation: typing.Sequence[int] | None = None,
    alpha: float | None = None,
    slocation: typing.Sequence[tuple[FREQ, SIGMA]] | typing.Sequence[float] | None = None,
    ssx: typing.Sequence[tuple[FREQ, SIGMA]] | typing.Sequence[float] | None = None,
    ssy: typing.Sequence[tuple[FREQ, SIGMA]] | typing.Sequence[float] | None = None,
    sst: typing.Sequence[tuple[FREQ, SIGMA]] | typing.Sequence[float] | None = None,
    ssystem: typing.Literal[0, 1] = 0,
    planes: int | typing.Sequence[int] | None = None,
    backend: backendT | None = None,
) -> ConstantFormatVideoNode: ...
