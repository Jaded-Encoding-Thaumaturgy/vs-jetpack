from __future__ import annotations
from jetpytools import CustomValueError, DependencyNotFoundError, KwargsT, inject_self
from inspect import Signature
from math import ceil

from vstools import ConstantFormatVideoNode, vs, core
from typing import Any, Protocol
from .abstract import Kernel

from typing import TypeVar


__all__ = [
    'CustomKernel'
]


class _kernel_func(Protocol):
    def __call__(self, *, x: float) -> float:
        ...


class CustomKernel(Kernel):
    def kernel(self, *, x: float) -> float:
        raise NotImplementedError

    def _modify_kernel_func(self, kwargs: KwargsT) -> tuple[_kernel_func, float]:
        blur = float(kwargs.pop('blur', 1.0))
        taps = int(kwargs.pop('taps', self.kernel_radius))
        support = taps * blur

        if blur != 1.0:
            def kernel(*, x: float) -> float:
                return self.kernel(x=x / blur)

            return kernel, support

        return self.kernel, support

    @inject_self
    def scale_function(  # type: ignore[override]
        self, clip: vs.VideoNode, width: int | None = None, height: int | None = None, *args: Any, **kwargs: Any
    ) -> vs.VideoNode:

        if not hasattr(core, 'resize2'):
            raise DependencyNotFoundError(
                self.__class__, 'resize2', 'Missing dependency \'resize2\'! '
                'You can find it here: https://github.com/Jaded-Encoding-Thaumaturgy/vapoursynth-resize2'
            )

        kernel, support = self._modify_kernel_func(kwargs)

        clean_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in Signature.from_callable(self._modify_kernel_func).parameters.keys()
        }

        return core.resize2.Custom(clip, kernel, ceil(support), width, height, *args, **clean_kwargs)

    @inject_self
    def resample_function(  # type: ignore[override]
        self, clip: vs.VideoNode, width: int | None = None, height: int | None = None, *args: Any, **kwargs: Any
    ) -> ConstantFormatVideoNode:
        return self.scale_function(clip, width, height, *args, **kwargs)  # type: ignore[return-value]

    @inject_self
    def descale_function(  # type: ignore[override]
        self, clip: vs.VideoNode, width: int, height: int, *args: Any, **kwargs: Any
    ) -> ConstantFormatVideoNode:
        kernel, support = self._modify_kernel_func(kwargs)

        clean_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in Signature.from_callable(self._modify_kernel_func).parameters.keys()
        }

        try:
            return core.descale.Decustom(clip, width, height, kernel, ceil(support), *args, **clean_kwargs)
        except vs.Error as e:
            if 'Output dimension must be' in str(e):
                raise CustomValueError(
                    f'Output dimension ({width}x{height}) must be less than or equal to '
                    f'input dimension ({clip.width}x{clip.height}).', self.__class__
                )

            raise CustomValueError(e, self.__class__)

    def get_params_args(
        self, is_descale: bool, clip: vs.VideoNode, width: int | None = None, height: int | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        args = super().get_params_args(is_descale, clip, width, height, **kwargs)

        if not is_descale:
            for key in ('border_handling', 'ignore_mask'):
                args.pop(key, None)

        return args


CustomKernelT = TypeVar('CustomKernelT', bound=CustomKernel)
