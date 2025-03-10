
from abc import abstractmethod
from ctypes import c_void_p
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Literal, NamedTuple
from weakref import ReferenceType

from ._nodes import AudioNode, VideoNode

__all__ = [
    # Environment SubSystem
    'Environment', 'EnvironmentData',

    'EnvironmentPolicy',

    'EnvironmentPolicyAPI',
    'register_policy', 'has_policy',
    'register_on_destroy', 'unregister_on_destroy',

    'get_current_environment',

    'VideoOutputTuple',
    'clear_output', 'clear_outputs', 'get_outputs', 'get_output',

    # Inspection API [UNSTABLE API]
    # '_try_enable_introspection'
]


###
# VapourSynth Environment SubSystem


class EnvironmentData: ...


class EnvironmentPolicy:
    def on_policy_registered(self, special_api: EnvironmentPolicyAPI) -> None: ...

    def on_policy_cleared(self) -> None: ...

    @abstractmethod
    def get_current_environment(self) -> EnvironmentData | None: ...

    @abstractmethod
    def set_environment(self, environment: EnvironmentData | None) -> EnvironmentData | None: ...

    def is_alive(self, environment: EnvironmentData) -> bool: ...


class EnvironmentPolicyAPI:
    def wrap_environment(self, environment_data: EnvironmentData) -> Environment: ...

    def create_environment(self, flags: int = 0) -> EnvironmentData: ...

    def set_logger(self, env: EnvironmentData, logger: Callable[[int, str], None]) -> None: ...

    def get_vapoursynth_api(self, version: int) -> c_void_p: ...

    def get_core_ptr(self, environment_data: EnvironmentData) -> c_void_p: ...

    def destroy_environment(self, env: EnvironmentData) -> None: ...

    def unregister_policy(self) -> None: ...


def register_policy(policy: EnvironmentPolicy) -> None:
    ...


if not TYPE_CHECKING:
    def _try_enable_introspection(version: int = None): ...


def has_policy() -> bool: ...


def register_on_destroy(callback: Callable[..., None]) -> None: ...


def unregister_on_destroy(callback: Callable[..., None]) -> None: ...


class Environment:
    env: ReferenceType[EnvironmentData]

    @property
    def alive(self) -> bool: ...

    @property
    def single(self) -> bool: ...

    @classmethod
    def is_single(cls) -> bool: ...

    @property
    def env_id(self) -> int: ...

    @property
    def active(self) -> bool: ...

    def copy(self) -> 'Environment': ...

    def use(self) -> ContextManager[None]: ...

    def __eq__(self, other: 'Environment') -> bool: ...  # type: ignore[override]

    def __repr__(self) -> str: ...


def get_current_environment() -> Environment: ...


class Local:
    def __getattr__(self, key: str) -> Any: ...
    
    # Even though object does have set/del methods, typecheckers will treat them differently
    # when they are not explicit; for example by raising a member not found warning.

    def __setattr__(self, key: str, value: Any) -> None: ...
    
    def __delattr__(self, key: str) -> None: ...


class VideoOutputTuple(NamedTuple):
    clip: 'VideoNode'
    alpha: 'VideoNode' | None
    alt_output: Literal[0, 1, 2]


def clear_output(index: int = 0) -> None: ...


def clear_outputs() -> None: ...


def get_outputs() -> MappingProxyType[int, VideoOutputTuple | 'AudioNode']: ...


def get_output(index: int = 0) -> VideoOutputTuple | 'AudioNode':...
