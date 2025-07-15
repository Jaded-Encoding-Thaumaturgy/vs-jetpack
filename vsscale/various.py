from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from vsexprtools import ExprOp, combine, norm_expr
from vskernels import Catrom, KernelLike, Scaler, ScalerLike
from vsmasktools import ringing_mask
from vsrgtools.rgtools import Repair
from vstools import (
    ConstantFormatVideoNode,
    CustomOverflowError,
    check_ref_clip,
    scale_delta,
    vs,
)

from .generic import GenericScaler

__all__ = ["ClampScaler"]


class ClampScaler(GenericScaler):
    """
    Clamp a reference Scaler.
    """

    def __init__(
        self,
        base_scaler: ScalerLike,
        reference: ScalerLike | vs.VideoNode,
        strength: int = 80,
        overshoot: float | None = None,
        undershoot: float | None = None,
        limit: Repair.Mode | bool = True,
        operator: Literal[ExprOp.MAX, ExprOp.MIN] | None = ExprOp.MIN,
        masked: bool = True,
        *,
        kernel: KernelLike = Catrom,
        scaler: ScalerLike | None = None,
        shifter: KernelLike | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            base_scaler: Scaler to clamp.
            reference: Reference Scaler used to clamp base_scaler
            strength: Strength of clamping. Default to 80. Must be between 0 and 100 (exclusive)
            overshoot: Overshoot threshold within the 0.0 - 1.0 range.
            undershoot: Undershoot threshold within the 0.0 - 1.0 range.
            limit: Whether to use under/overshoot limit (True) or a reference repaired clip for limiting.
            operator: Whether to take the brightest or darkest pixels in the merge. Defaults to ExprOp.MIN.
            masked: Whether to mask with a ringing mask or not. Defaults to True
            kernel: Base kernel to be used for certain scaling/shifting/resampling operations. Defaults to Catrom.
            scaler: Scaler used for scaling operations. Defaults to kernel.
            shifter: Kernel used for shifting operations. Defaults to kernel.
        """
        self.base_scaler = Scaler.ensure_obj(base_scaler, self.__class__)

        self.reference: Scaler | vs.VideoNode

        if not isinstance(reference, vs.VideoNode):
            self.reference = Scaler.ensure_obj(reference, self.__class__)
        else:
            self.reference = reference

        if not 0 < strength < 100:
            raise CustomOverflowError("`strength` must be between 0 and 100 (exclusive)!", self.__class__)

        self.strength = strength

        if overshoot is None:
            self.overshoot = self.strength / 100
        else:
            self.overshoot = overshoot

        if undershoot is None:
            self.undershoot = self.overshoot
        else:
            self.undershoot = undershoot

        self.limit = limit
        self.operator = operator
        self.masked = masked

        super().__init__(None, kernel=kernel, scaler=scaler, shifter=shifter, **kwargs)

    def scale(
        self,
        clip: vs.VideoNode,
        width: int | None = None,
        height: int | None = None,
        shift: tuple[float, float] = (0, 0),
        **kwargs: Any,
    ) -> ConstantFormatVideoNode:
        width, height = self._wh_norm(clip, width, height)

        base = self.base_scaler.scale(clip, width, height, shift, **kwargs)

        if isinstance(self.reference, vs.VideoNode):
            smooth = self.reference

            if shift != (0, 0):
                smooth = self.kernel.shift(smooth, shift)
        else:
            smooth = self.reference.scale(clip, width, height, shift)

        check_ref_clip(base, smooth)

        if TYPE_CHECKING:
            from vstools import check_variable_format

            assert check_variable_format(base, self.__class__)
            assert check_variable_format(smooth, self.__class__)

        merge_weight = self.strength / 100

        if self.limit is True:
            expression = "x {merge_weight} * y {ref_weight} * + a {undershoot} - z {overshoot} + clip"

            merged = norm_expr(
                [base, smooth, smooth.std.Maximum(), smooth.std.Minimum()],
                expression,
                merge_weight=merge_weight,
                ref_weight=1.0 - merge_weight,
                undershoot=scale_delta(self.undershoot, 32, clip),
                overshoot=scale_delta(self.overshoot, 32, clip),
                func=self.__class__,
            )
        else:
            merged = smooth.std.Merge(base, merge_weight)

            if isinstance(self.limit, Repair.Mode):
                merged = self.limit(merged, smooth)

        if self.operator is not None:
            merge2 = combine([smooth, base], self.operator)

            merged = merged.std.MaskedMerge(merge2, ringing_mask(smooth)) if self.masked else merge2
        elif self.masked:
            merged = merged.std.MaskedMerge(smooth, ringing_mask(smooth))

        return merged

    @Scaler.cached_property
    def kernel_radius(self) -> int:
        if not isinstance(self.reference, vs.VideoNode):
            return max(self.reference.kernel_radius, self.base_scaler.kernel_radius)
        return self.base_scaler.kernel_radius
