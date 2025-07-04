from inspect import Signature
from typing import Iterator

from ._typings import _VapourSynthMapValue
from ._version import PluginVersion

__all__ = [
    "FuncData",
    "Func",
    "FramePtr",
    "Plugin",
    "Function",
]

class FuncData:
    def __call__(self, **kwargs: _VapourSynthMapValue) -> _VapourSynthMapValue: ...

class Func:
    def __call__(self, **kwargs: _VapourSynthMapValue) -> _VapourSynthMapValue: ...

class FramePtr: ...

class Function:
    plugin: Plugin
    name: str
    signature: str
    return_signature: str

    def __call__(self, *args: _VapourSynthMapValue, **kwargs: _VapourSynthMapValue) -> _VapourSynthMapValue: ...
    @property
    def __signature__(self) -> Signature: ...

class Plugin:
    identifier: str
    namespace: str
    name: str

    def __getattr__(self, name: str) -> Function: ...
    def functions(self) -> Iterator[Function]: ...
    @property
    def version(self) -> PluginVersion: ...
    @property
    def plugin_path(self) -> str: ...
