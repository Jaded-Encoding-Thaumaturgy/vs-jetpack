from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol

from vskernels import Catrom, Kernel, KernelLike, Scaler, ScalerLike
from vstools import ConstantFormatVideoNode, MatrixT, check_variable, plane, vs

__all__ = [
    "BaseGenericScaler",
    "GenericScaler",
]


class _GeneriScaleNoShift(Protocol):
    def __call__(
        self, clip: ConstantFormatVideoNode, width: int, height: int, *args: Any, **kwargs: Any
    ) -> ConstantFormatVideoNode: ...


class _GeneriScaleWithShift(Protocol):
    def __call__(
        self,
        clip: ConstantFormatVideoNode,
        width: int,
        height: int,
        shift: tuple[float, float],
        *args: Any,
        **kwargs: Any,
    ) -> ConstantFormatVideoNode: ...


def _func_no_op(clip: ConstantFormatVideoNode, *args: Any, **kwargs: Any) -> ConstantFormatVideoNode:
    return clip


class BaseGenericScaler(Scaler, ABC):
    """
    Generic Scaler base class.
    Inherit from this to create more complex scalers with built-in utils.
    """

    def __init__(
        self,
        *,
        kernel: KernelLike = Catrom,
        scaler: ScalerLike | None = None,
        shifter: KernelLike | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the BaseGenericScaler.

        Args:
            kernel: Base kernel to be used for certain scaling/shifting/resampling operations. Defaults to Catrom.
            scaler: Scaler used for scaling operations. Defaults to kernel.
            shifter: Kernel used for shifting operations. Defaults to kernel.
        """
        self.kernel = Kernel.ensure_obj(kernel, self.__class__)
        self.scaler = Scaler.ensure_obj(scaler or self.kernel, self.__class__)
        self.shifter = Kernel.ensure_obj(shifter or self.kernel, self.__class__)

        super().__init__(**kwargs)

    @abstractmethod
    def scale(
        self,
        clip: vs.VideoNode,
        width: int | None = None,
        height: int | None = None,
        shift: tuple[float, float] = (0, 0),
        **kwargs: Any,
    ) -> ConstantFormatVideoNode:
        assert check_variable(clip, self.__class__)

        width, height = self._wh_norm(clip, width, height)

        return self._finish_scale(clip, clip, width, height, shift)

    def _finish_scale(
        self,
        clip: ConstantFormatVideoNode,
        input_clip: ConstantFormatVideoNode,
        width: int,
        height: int,
        shift: tuple[float, float] = (0, 0),
        matrix: MatrixT | None = None,
        copy_props: bool = False,
    ) -> ConstantFormatVideoNode:
        if input_clip.format.num_planes == 1:
            clip = plane(clip, 0)

        if (clip.width, clip.height) != (width, height):
            clip = self.scaler.scale(clip, width, height)  # type: ignore[assignment]

        if shift != (0, 0):
            clip = self.shifter.shift(clip, shift)

        if clip.format.id != input_clip.format.id:
            clip = self.kernel.resample(clip, input_clip, matrix)

        if copy_props:
            return vs.core.std.CopyFrameProps(clip, input_clip)

        return clip


class GenericScaler(BaseGenericScaler, partial_abstract=True):
    """
    Generic Scaler class
    """

    def __init__(
        self,
        func: _GeneriScaleNoShift | _GeneriScaleWithShift | None = None,
        *,
        kernel: KernelLike = Catrom,
        scaler: ScalerLike | None = None,
        shifter: KernelLike | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Apply an arbitrary scaling function.

        Args:
            func: The scaling function to apply. Can either be a function without shifting or one that includes shifting
                logic.
            kernel: Base kernel to be used for certain scaling/shifting/resampling operations. Defaults to Catrom.
            scaler: Scaler used for scaling operations. Defaults to kernel.
            shifter: Kernel used for shifting operations. Defaults to kernel.
        """
        self.func = _func_no_op if func is None else func
        super().__init__(kernel=kernel, scaler=scaler, shifter=shifter, **kwargs)

    def scale(
        self,
        clip: vs.VideoNode,
        width: int | None = None,
        height: int | None = None,
        shift: tuple[float, float] = (0, 0),
        **kwargs: Any,
    ) -> ConstantFormatVideoNode:
        assert check_variable(clip, self.__class__)

        width, height = self._wh_norm(clip, width, height)

        if shift != (0, 0):
            output = self.func(clip, width, height, shift, **kwargs)
        else:
            output = self.func(clip, width, height, **kwargs)

        return self._finish_scale(output, clip, width, height, shift)
