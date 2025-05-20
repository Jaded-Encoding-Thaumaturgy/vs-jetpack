from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import IntFlag, auto
from typing import Any, Sequence

from jetpytools import inject_self
from typing_extensions import Self

from vskernels import LeftShift, Scaler, TopShift
from vstools import (
    ChromaLocation, ConstantFormatVideoNode, VSFunctionAllArgs, VSFunctionNoArgs, check_variable, core, fallback,
    normalize_seq, vs, vs_object
)


class AADirection(IntFlag):
    VERTICAL = auto()
    HORIZONTAL = auto()
    BOTH = VERTICAL | HORIZONTAL


@dataclass(kw_only=True)
class Deinterlacer(vs_object, ABC):
    tff: bool = False
    double_rate: bool = True
    transpose_first: bool = False

    @property
    @abstractmethod
    def _deinterlacer_function(self) -> VSFunctionAllArgs[vs.VideoNode, ConstantFormatVideoNode]:
        """Get the plugin function"""

    @abstractmethod
    def _interpolate(self, clip: vs.VideoNode, tff: bool, dh: bool, **kwargs: Any) -> ConstantFormatVideoNode:
        """
        Performs deinterlacing if dh is False or doubling if dh is True.
        Should handle tff to field if needed and should add the kwargs from `get_deint_args`
        """

    @abstractmethod
    def get_deint_args(self, **kwargs: Any) -> dict[str, Any]:
        return kwargs

    def deinterlace(self, clip: vs.VideoNode, **kwargs: Any) -> ConstantFormatVideoNode:
        return self._interpolate(clip, self.tff, False, **kwargs)

    def antialias(
        self, clip: vs.VideoNode, direction: AADirection = AADirection.BOTH, **kwargs: Any
    ) -> ConstantFormatVideoNode:
        assert check_variable(clip, self.antialias)

        for y in sorted((aa_dir for aa_dir in AADirection), key=lambda x: x.value, reverse=self.transpose_first):
            if direction in (y, AADirection.BOTH):
                if y == AADirection.HORIZONTAL:
                    clip = self.transpose(clip)

                clip = self.deinterlace(clip, **kwargs)

                if self.double_rate:
                    clip = core.std.Merge(clip[::2], clip[1::2])

                if y == AADirection.HORIZONTAL:
                    clip = self.transpose(clip)

        return clip

    def transpose(self, clip: vs.VideoNode) -> ConstantFormatVideoNode:
        return clip.std.Transpose()

    def copy(self, **kwargs: Any) -> Self:
        """Returns a new Antialiaser class replacing specified fields with new values"""
        return replace(self, **kwargs)


class SuperSampler(Deinterlacer, Scaler, ABC):
    # TODO: Change this when #94 is merged 
    @inject_self.cached
    def scale(
        self,
        clip: vs.VideoNode,
        width: int | None = None,
        height: int | None = None,
        shift: tuple[TopShift, LeftShift] = (0, 0),
        **kwargs: Any
    ) -> ConstantFormatVideoNode:
        assert check_variable(clip, self.scale)

        dest_dimensions = list(self._wh_norm(clip, width, height))
        sy, sx = shift

        cloc = list(ChromaLocation.get_offsets(clip, ChromaLocation.from_video(clip)))
        subsampling = [2**clip.format.subsampling_w, 2**clip.format.subsampling_h]

        nshift: list[list[float]] = [
            normalize_seq(sx, clip.format.num_planes),
            normalize_seq(sy, clip.format.num_planes)
        ]

        if not self.transpose_first:
            dest_dimensions.reverse()
            cloc.reverse()
            subsampling.reverse()
            nshift.reverse()

        for x, dim in enumerate(dest_dimensions):
            is_width = not x and self.transpose_first or not self.transpose_first and x

            while (clip.width if is_width else clip.height) < dim:
                delta = max(nshift[x], key=lambda y: abs(y))
                tff = False if delta < 0 else True if delta > 0 else self.tff

                for y in range(clip.format.num_planes):
                    if not y:
                        # TODO: Change this when #94 is merged 
                        nshift[x][y] = (nshift[x][y] + (-0.25 if tff else 0.25)) * 2
                    else:
                        nshift[x][y] = (nshift[x][y] + (-0.25 if tff else 0.25) * subsampling[x]) * 2 - cloc[x]

                if is_width:
                    clip = self.transpose(clip)

                clip = self._interpolate(clip, tff, True, **kwargs)

                if is_width:
                    clip = self.transpose(clip)

        if not self.transpose_first:
            nshift.reverse()

        # TODO: Change this when #94 is merged 
        return clip.fmtc.resample(width, height, nshift[0], nshift[1])


@dataclass
class NNEDI3(SuperSampler, Deinterlacer):
    nsize: int | None = None
    nns: int | None = None
    qual: int | None = None
    etype: int | None = None
    pscrn: int | None = None
    opencl: bool = False

    @property
    def _deinterlacer_function(self) -> VSFunctionAllArgs[vs.VideoNode, ConstantFormatVideoNode]:
        return core.lazy.sneedif.NNEDI3 if self.opencl else core.lazy.znedi3.nnedi3

    def _interpolate(self, clip: vs.VideoNode, tff: bool, dh: bool, **kwargs: Any) -> ConstantFormatVideoNode:
        field = int(tff) + (int(self.double_rate) * 2)

        return self._deinterlacer_function(clip, field, dh, **self.get_deint_args(**kwargs))

    def get_deint_args(self, **kwargs: Any) -> dict[str, Any]:
        return dict(
            nsize=self.nsize,
            nns=self.nns,
            qual=self.qual,
            etype=self.etype,
            pscrn=self.pscrn
        ) | kwargs

    # TODO: Change this when #94 is merged 
    @inject_self.cached.property
    def kernel_radius(self) -> int:
        match self.nsize:
            case 1 | 5:
                return 16
            case 2 | 6:
                return 32
            case 3:
                return 48
            case _:
                return 8


@dataclass
class EEDI2(SuperSampler, Deinterlacer):
    mthresh: int | None = None
    lthresh: int | None = None
    vthresh: int | None = None
    estr: int | None = None
    dstr: int | None = None
    maxd: int | None = None
    map: int | None = None
    nt: int | None = None
    pp: int | None = None
    cuda: bool = False

    @property
    def _deinterlacer_function(self) -> VSFunctionAllArgs[vs.VideoNode, ConstantFormatVideoNode]:
        return core.lazy.eedi2cuda.EEDI2 if self.cuda else core.lazy.eedi2.EEDI2

    def _interpolate(self, clip: vs.VideoNode, tff: bool, dh: bool, **kwargs: Any) -> ConstantFormatVideoNode:
        field = int(tff)

        if not dh:
            field += int(self.double_rate) * 2
            clip = clip.std.SeparateFields(tff)

            if not self.double_rate:
                clip = clip[::2]

        return self._deinterlacer_function(clip, field, **self.get_deint_args(**kwargs))

    def get_deint_args(self, **kwargs: Any) -> dict[str, Any]:
        return dict(
            mthresh=self.mthresh,
            lthresh=self.lthresh,
            vthresh=self.vthresh,
            estr=self.estr,
            dstr=self.dstr,
            maxd=self.maxd,
            map=self.map,
            nt=self.nt,
            pp=self.pp
        ) | kwargs

    # TODO: Change this when #94 is merged 
    @inject_self.cached.property
    def kernel_radius(self) -> int:
        return fallback(self.maxd, 24)


@dataclass
class EEDI3(SuperSampler, Deinterlacer):
    alpha: float | None = None
    beta: float | None = None
    gamma: float | None = None
    nrad: int | None = None
    mdis: int | None = None
    ucubic: bool | None = None
    cost3: bool | None = None
    vcheck: int | None = None
    vthresh: Sequence[float | None] | None = None
    sclip: vs.VideoNode | VSFunctionNoArgs[vs.VideoNode, ConstantFormatVideoNode] | None = None
    mclip: vs.VideoNode | VSFunctionNoArgs[vs.VideoNode, ConstantFormatVideoNode] | None = None
    opencl: bool = False

    @property
    def _deinterlacer_function(self) -> VSFunctionAllArgs[vs.VideoNode, ConstantFormatVideoNode]:
        return core.lazy.eedi3m.EEDI3CL if self.opencl else core.lazy.eedi3m.EEDI3

    def _interpolate(self, clip: vs.VideoNode, tff: bool, dh: bool, **kwargs: Any) -> ConstantFormatVideoNode:
        field = int(tff) if dh else int(tff) + (int(self.double_rate) * 2)

        kwargs = self.get_deint_args(**kwargs)

        if callable(self.sclip):
            kwargs.update(sclip=self.sclip(clip))

        if callable(self.mclip):
            kwargs.update(mclip=self.mclip(clip))

        mult = (0 if dh else int(self.double_rate)) + 1

        if sclip := kwargs.get('sclip'):
            if sclip.num_frames * 2 == clip.num_frames * mult:
                kwargs.update(sclip=core.std.Interleave([sclip] * 2))

        if mclip := kwargs.get('mclip'):
            if mclip.num_frames * 2 == clip.num_frames * mult:
                kwargs.update(mclip=core.std.Interleave([mclip] * 2))

        return self._deinterlacer_function(clip, field, dh, **kwargs)

    def transpose(self, clip: vs.VideoNode) -> ConstantFormatVideoNode:
        if isinstance(self.sclip, vs.VideoNode):
            self.sclip = self.sclip.std.Transpose()

        if isinstance(self.mclip, vs.VideoNode):
            self.mclip = self.mclip.std.Transpose()

        return super().transpose(clip)

    def get_deint_args(self, **kwargs: Any) -> dict[str, Any]:
        self.vthresh = normalize_seq(self.vthresh, 3)

        return dict(
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            nrad=self.nrad,
            mdis=self.mdis,
            ucubic=self.ucubic,
            cost3=self.cost3,
            vcheck=self.vcheck,
            vthresh0=self.vthresh[0],
            vthresh1=self.vthresh[1],
            vthresh2=self.vthresh[2],
            sclip=self.sclip,
            mclip=self.mclip
        ) | kwargs

    # TODO: Change this when #94 is merged 
    @inject_self.cached.property
    def kernel_radius(self) -> int:
        return fallback(self.mdis, 20)

    def __vs_del__(self, core_id: int) -> None:
        self.sclip = None
        self.mclip = None


@dataclass
class SANGNOM(SuperSampler, Deinterlacer):
    aa: int | Sequence[int] | None = None

    @property
    def _deinterlacer_function(self) -> VSFunctionAllArgs[vs.VideoNode, ConstantFormatVideoNode]:
        return core.lazy.sangnom.SangNom

    def _interpolate(self, clip: vs.VideoNode, tff: bool, dh: bool, **kwargs: Any) -> ConstantFormatVideoNode:
        if self.double_rate and not dh:
            order = 0
            clip = clip.std.SeparateFields(tff).std.DoubleWeave(tff)
        else:
            order = 1 if tff else 2

        return self._deinterlacer_function(clip, order, dh, **self.get_deint_args(**kwargs))

    def get_deint_args(self, **kwargs: Any) -> dict[str, Any]:
        return dict(aa=self.aa) | kwargs

    _static_kernel_radius = 3
