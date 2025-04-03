from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from vsexprtools import ExprOp, combine, norm_expr
from vskernels import ScalerT
from vsmasktools import ringing_mask
from vsrgtools import RepairMode, repair
from vstools import CustomOverflowError, check_ref_clip, inject_self, scale_delta, vs


from .helpers import GenericScaler

__all__ = [
    'ClampScaler',
]


@dataclass
class ClampScaler(GenericScaler):
    """Clamp a reference Scaler."""

    base_scaler: ScalerT
    """Scaler to clamp."""

    reference: ScalerT | vs.VideoNode
    """Reference Scaler used to clamp base_scaler"""

    strength: int = 80
    """Strength of clamping."""

    overshoot: float | None = None
    """Overshoot threshold."""

    undershoot: float | None = None
    """Undershoot threshold."""

    limit: RepairMode | bool = True
    """Whether to use under/overshoot limit (True) or a reference repaired clip for limiting."""

    operator: Literal[ExprOp.MAX, ExprOp.MIN] | None = ExprOp.MIN
    """Whether to take the brightest or darkest pixels in the merge."""

    masked: bool = True
    """Whether to mask with a ringing mask or not."""

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.strength >= 100:
            raise CustomOverflowError('strength can\'t be more or equal to 100!', self.__class__)
        elif self.strength <= 0:
            raise CustomOverflowError('strength can\'t be less or equal to 0!', self.__class__)

        if self.overshoot is None:
            self.overshoot = self.strength / 100
        if self.undershoot is None:
            self.undershoot = self.overshoot

        self._base_scaler = self.ensure_scaler(self.base_scaler)
        self._reference = None if isinstance(self.reference, vs.VideoNode) else self.ensure_scaler(self.reference)

    @inject_self
    def scale(  # type: ignore
        self, clip: vs.VideoNode, width: int | None = None, height: int | None = None,
        shift: tuple[float, float] = (0, 0), *, smooth: vs.VideoNode | None = None, **kwargs: Any
    ) -> vs.VideoNode:
        width, height = self._wh_norm(clip, width, height)

        assert (self.undershoot or self.undershoot == 0) and (self.overshoot or self.overshoot == 0)

        ref = self._base_scaler.scale(clip, width, height, shift, **kwargs)

        if isinstance(self.reference, vs.VideoNode):
            smooth = self.reference

            if shift != (0, 0):
                smooth = self._kernel.shift(smooth, shift)
        else:
            assert self._reference
            smooth = self._reference.scale(clip, width, height, shift)

        assert smooth

        check_ref_clip(ref, smooth)

        merge_weight = self.strength / 100

        if self.limit is True:
            expression = 'x {merge_weight} * y {ref_weight} * + a {undershoot} - z {overshoot} + clip'

            merged = norm_expr(
                [ref, smooth, smooth.std.Maximum(), smooth.std.Minimum()],
                expression, merge_weight=merge_weight, ref_weight=1.0 - merge_weight,
                undershoot=scale_delta(self.undershoot, 32, clip),
                overshoot=scale_delta(self.overshoot, 32, clip),
                func=self.__class__
            )
        else:
            merged = smooth.std.Merge(ref, merge_weight)

            if isinstance(self.limit, RepairMode):
                merged = repair(merged, smooth, self.limit)

        if self.operator is not None:
            merge2 = combine([smooth, ref], self.operator)

            if self.masked:
                merged = merged.std.MaskedMerge(merge2, ringing_mask(smooth))
            else:
                merged = merge2
        elif self.masked:
            merged.std.MaskedMerge(smooth, ringing_mask(smooth))

        return merged

    @property
    def kernel_radius(self) -> int:  # type: ignore[override]
        if self._reference:
            return max(self._reference.kernel_radius, self._base_scaler.kernel_radius)
        return self._base_scaler.kernel_radius
