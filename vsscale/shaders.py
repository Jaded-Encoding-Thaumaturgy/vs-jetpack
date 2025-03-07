from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Any, overload

from vstools import (
    MISSING, CustomRuntimeError, FileWasNotFoundError, MissingT, core, expect_bits, get_user_data_dir, get_video_format,
    inject_self, join, vs
)

from .base import ShaderFileBase, ShaderFileCustom
from .helpers import GenericScaler

__all__ = [
    'PlaceboShader',

    'ShaderFile'
]


class PlaceboShaderMeta(GenericScaler):
    shader_file: str | Path | ShaderFile

    _static_kernel_radius = 2


@dataclass
class PlaceboShaderBase(PlaceboShaderMeta):
    """Base placebo shader class."""

    chroma_loc: int | None = field(default=None, kw_only=True)
    matrix: int | None = field(default=None, kw_only=True)
    trc: int | None = field(default=None, kw_only=True)
    linearize: int | None = field(default=None, kw_only=True)
    sigmoidize: int | None = field(default=None, kw_only=True)
    sigmoid_center: float | None = field(default=None, kw_only=True)
    sigmoid_slope: float | None = field(default=None, kw_only=True)
    antiring: float | None = field(default=None, kw_only=True)
    filter_shader: str | None = field(default=None, kw_only=True)
    clamp: float | None = field(default=None, kw_only=True)
    blur: float | None = field(default=None, kw_only=True)
    taper: float | None = field(default=None, kw_only=True)
    radius: float | None = field(default=None, kw_only=True)
    param1: float | None = field(default=None, kw_only=True)
    param2: float | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()

        if not hasattr(self, 'shader_file'):
            raise CustomRuntimeError('You must specify a "shader_file"!', self.__class__)

    @inject_self
    def scale(  # type: ignore
        self, clip: vs.VideoNode, width: int | None = None, height: int | None = None,
        shift: tuple[float, float] = (0, 0), **kwargs: Any
    ) -> vs.VideoNode:
        width, height = self._wh_norm(clip, width, height)

        output, _ = expect_bits(clip, 16)

        fmt = get_video_format(output)

        if fmt.num_planes == 1:
            if width > output.width or height > output.height:
                output = output.resize.Point(format=vs.YUV444P16)
            else:
                for div in (4, 2):
                    if width % div == 0 and height % div == 0:
                        blank = core.std.BlankClip(output, output.width // div, output.height // div, vs.GRAY16)
                        break
                else:
                    blank = output.std.BlankClip(vs.GRAY16)

                output = join(output, blank, blank)

        kwargs |= {
            'shader': str(
                self.shader_file()
                if isinstance(self.shader_file, ShaderFile) else
                ShaderFile.CUSTOM(self.shader_file)  # type: ignore
            ),
            'chroma_loc': self.chroma_loc, 'matrix': self.matrix,
            'trc': self.trc, 'linearize': self.linearize,
            'sigmoidize': self.sigmoidize, 'sigmoid_center': self.sigmoid_center, 'sigmoid_slope': self.sigmoid_slope,
            'antiring': self.antiring, 'filter': self.filter_shader, 'clamp': self.clamp,
            'blur': self.blur, 'taper': self.taper, 'radius': self.radius,
            'param1': self.param1, 'param2': self.param2,
        } | kwargs | {
            'width': output.width * ceil(width / output.width),
            'height': output.height * ceil(height / output.height)
        }

        if not kwargs['filter']:
            kwargs['filter'] = 'box' if fmt.num_planes == 1 else 'ewa_lanczos'

        if not Path(kwargs['shader']).exists():
            try:
                kwargs['shader'] = str(ShaderFile.CUSTOM(kwargs['shader']))  # type: ignore
            except FileWasNotFoundError:
                ...

        output = output.placebo.Shader(**kwargs)

        return self._finish_scale(output, clip, width, height, shift)


@dataclass
class PlaceboShader(PlaceboShaderBase):
    shader_file: str | Path


class ShaderFile(ShaderFileBase):
    """Default shader files shipped with vsscale."""

    if not TYPE_CHECKING:
        CUSTOM = 'custom'

    @overload
    def __call__(self) -> Path:
        ...

    @overload
    def __call__(self: ShaderFileCustom, file_name: str | Path) -> Path:  # type: ignore
        ...

    def __call__(self, file_name: str | Path | MissingT = MISSING) -> Path:
        """Get a path from the shader member, name or path."""

        if self is not ShaderFile.CUSTOM:
            file_name = self.value

        if file_name is MISSING:
            raise TypeError("ShaderFile.__call__() missing 1 required positional argument: 'file_name'")

        file_name, cwd = Path(file_name), Path.cwd()

        assets_dirs = [
            file_name,
            cwd / file_name,
            cwd / '.shaders' / file_name,
            cwd / '_shaders' / file_name,
            cwd / '.assets' / file_name,
            cwd / '_assets' / file_name
        ]

        for asset_dir in assets_dirs:
            if asset_dir.is_file():
                return asset_dir

        mpv_dir = get_user_data_dir().parent / 'Roaming' / 'mpv' / 'shaders' / file_name

        if mpv_dir.is_file():
            return mpv_dir

        raise FileWasNotFoundError(f'"{file_name}" could not be found!', str(ShaderFile.CUSTOM))