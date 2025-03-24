from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import TYPE_CHECKING, Any, Literal

from vstools import ConstantFormatVideoNode, CustomValueError, core, inject_self, vs, vs_object

from ..abstract import Antialiaser, DoubleRater, SingleRater, SuperSampler, _Antialiaser
from . import nnedi3

__all__ = [
    'Eedi3', 'Eedi3DR'
]


@dataclass
class EEDI3(_Antialiaser):
    alpha: float = 0.25
    beta: float = 0.5
    gamma: float = 40
    nrad: int = 2
    mdis: int = 20

    ucubic: bool = True
    cost3: bool = True
    vcheck: int = 2
    vthresh0: float = 32.0
    vthresh1: float = 64.0
    vthresh2: float = 4.0

    opt: int = 0
    device: int = -1
    opencl: bool = dc_field(default=False, kw_only=True)

    mclip: vs.VideoNode | None = None
    sclip_aa: type[Antialiaser] | Antialiaser | Literal[True] | vs.VideoNode | None = dc_field(
        default_factory=lambda: nnedi3.Nnedi3
    )

    # Class Variable
    _shift = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()

        self._sclip_aa: Antialiaser | Literal[True] | vs.VideoNode | None

        if (
            self.sclip_aa is not None
            and self.sclip_aa is not True
            and not isinstance(self.sclip_aa, (Antialiaser, vs.VideoNode))
        ):
            self._sclip_aa = self.sclip_aa()
        else:
            self._sclip_aa = self.sclip_aa

    def get_aa_args(self, clip: vs.VideoNode, **kwargs: Any) -> dict[str, Any]:
        args = dict(
            alpha=self.alpha, beta=self.beta, gamma=self.gamma,
            nrad=self.nrad, mdis=self.mdis,
            ucubic=self.ucubic, cost3=self.cost3,
            vcheck=self.vcheck,
            vthresh0=self.vthresh0, vthresh1=self.vthresh1, vthresh2=self.vthresh2
        ) | kwargs

        if self.opencl:
            args |= dict(device=self.device)
        elif self.mclip is not None or kwargs.get('mclip'):
            # opt=3 appears to always give reliable speed boosts if mclip is used.
            args |= dict(opt=kwargs.get('opt', self.opt) or 3)

        return args

    def interpolate(self, clip: vs.VideoNode, double_y: bool, **kwargs: Any) -> ConstantFormatVideoNode:
        aa_kwargs = self.get_aa_args(clip, **kwargs)
        aa_kwargs = self._handle_sclip(clip, double_y, aa_kwargs, **kwargs)

        if self.opencl:
            interpolated = core.eedi3m.EEDI3CL(clip, self.field, double_y or not self.drop_fields, **aa_kwargs)
        else:
            interpolated = core.eedi3m.EEDI3(clip, self.field, double_y or not self.drop_fields, **aa_kwargs)

        return self.shift_interpolate(clip, interpolated, double_y, **kwargs)

    def _handle_sclip(
        self, clip: vs.VideoNode, double_y: bool, aa_kwargs: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        if not self._sclip_aa or ('sclip' in kwargs and kwargs['sclip']):
            return aa_kwargs

        if self._sclip_aa is True or isinstance(self._sclip_aa, vs.VideoNode):
            if double_y:
                if self._sclip_aa is True:
                    raise CustomValueError("You can't pass sclip_aa=True when supersampling!", self.__class__)

                if (clip.width, clip.height) != (self._sclip_aa.width, self._sclip_aa.height):
                    raise CustomValueError(
                        f'The dimensions of sclip_aa ({self._sclip_aa.width}x{self._sclip_aa.height}) '
                        f'don\'t match the expected dimensions ({clip.width}x{clip.height})!',
                        self.__class__
                    )

            aa_kwargs.update(sclip=clip)

            return aa_kwargs

        sclip_args = self._sclip_aa.get_aa_args(clip, mclip=kwargs.get('mclip'))
        sclip_args.update(self._sclip_aa.get_ss_args(clip) if double_y else self._sclip_aa.get_sr_args(clip))

        aa_kwargs.update(sclip=self._sclip_aa.interpolate(clip, double_y or not self.drop_fields, **sclip_args))

        return aa_kwargs

    def __del__(self) -> None:
        if not TYPE_CHECKING:
            self.mclip = None


class Eedi3SS(EEDI3, SuperSampler):
    @inject_self.cached.property
    def kernel_radius(self) -> int:
        return self.nrad


class Eedi3SR(EEDI3, SingleRater, vs_object):
    _mclips: tuple[vs.VideoNode, vs.VideoNode] | None = None

    def get_sr_args(self, clip: vs.VideoNode, **kwargs: Any) -> dict[str, Any]:
        if not self.mclip:
            return {}

        if not self._mclips:
            self._mclips = (self.mclip, self.mclip.std.Transpose())

        if self.mclip.width == clip.width and self.mclip.height == clip.height:
            return dict(mclip=self._mclips[0])
        else:
            return dict(mclip=self._mclips[1])

    def __del__(self) -> None:
        if not TYPE_CHECKING:
            self._mclips = None
            self.mclip = 0

    def __vs_del__(self, core_id: int) -> None:
        self._mclips = None


class Eedi3DR(EEDI3, DoubleRater):
    ...


class Eedi3(Eedi3SR, Eedi3SS, Antialiaser):
    ...
