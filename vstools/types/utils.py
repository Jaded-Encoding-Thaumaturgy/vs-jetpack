from __future__ import annotations

from abc import ABC, ABCMeta
from functools import partial
from typing import TYPE_CHECKING, Any, Callable

from jetpytools import (
    KwargsNotNone,
    LinearRangeLut,
    Singleton,
    cachedproperty,
    classproperty,
    complex_hash,
    copy_signature,
    get_subclasses,
    inject_self,
    to_singleton,
)
from typing_extensions import Self

__all__ = [
    "KwargsNotNone",
    "LinearRangeLut",
    "Singleton",
    "VSDebug",
    "cachedproperty",
    "classproperty",
    "complex_hash",
    "copy_signature",
    "get_subclasses",
    "inject_self",
    "to_singleton",
    "vs_object",
]


class vs_object(ABC, metaclass=ABCMeta):  # noqa: N801
    """
    Special object that follows the lifecycle of the VapourSynth environment/core.

    If a special dunder is created, __vs_del__, it will be called when the environment is getting deleted.

    This is especially useful if you have to hold a reference to a VideoNode or Plugin/Function object
    in this object as you need to remove it for the VapourSynth core to be freed correctly.
    """

    __vsdel_partial_register: Callable[..., None]
    __vsdel_register: Callable[[int], None] | None = None

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        from ..utils.vs_proxy import core, register_on_creation

        try:
            self = super().__new__(cls, *args, **kwargs)
        except TypeError:
            self = super().__new__(cls)

        if hasattr(self, "__vs_del__"):

            def _register(core_id: int) -> None:
                self.__vsdel_partial_register = partial(self.__vs_del__, core_id)
                core.register_on_destroy(self.__vsdel_partial_register)

            # [un]register_on_creation/destroy will only hold a weakref to the object
            self.__vsdel_register = _register
            register_on_creation(self.__vsdel_register)

        return self

    def __post_init__(self) -> None: ...

    if TYPE_CHECKING:

        def __vs_del__(self, core_id: int) -> None:
            """
            Special dunder that will be called when a core is getting freed.
            """


class VSDebug(Singleton, init=True):
    """
    Special class that follows the VapourSynth lifecycle for debug purposes.
    """

    _print_func: Callable[..., None] = print

    def __init__(self, *, env_life: bool = True, core_fetch: bool = False, use_logging: bool = False) -> None:
        """
        Print useful debug information.

        Args:
            env_life: Print creation/destroy of VapourSynth environment.
            core_fetch: Print traceback of the code that led to the first concrete core fetch. Especially useful when
                trying to find the code path that is locking you into a EnvironmentPolicy.
        """

        from ..utils.vs_proxy import register_on_creation

        if use_logging:
            import logging

            VSDebug._print_func = logging.debug
        else:
            VSDebug._print_func = print

        if env_life:
            register_on_creation(VSDebug._print_env_live, True)

        if core_fetch:
            register_on_creation(VSDebug._print_stack, True)

    @staticmethod
    def _print_stack(core_id: int) -> None:
        raise Exception

    @staticmethod
    def _print_env_live(core_id: int) -> None:
        from ..utils.vs_proxy import core, register_on_destroy

        VSDebug._print_func(f"New core created with id: {core_id}")

        core.register_on_destroy(VSDebug._print_core_destroy, False)
        register_on_destroy(partial(VSDebug._print_destroy, core.env.env_id, core_id))

    @staticmethod
    def _print_destroy(env_id: int, core_id: int) -> None:
        VSDebug._print_func(f"Environment destroyed with id: {env_id}, current core id: {core_id}")

    @staticmethod
    def _print_core_destroy(_: int, core_id: int) -> None:
        VSDebug._print_func(f"Core destroyed with id: {core_id}")
