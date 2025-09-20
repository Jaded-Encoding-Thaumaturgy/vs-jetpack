from __future__ import annotations

from functools import partial
from typing import Callable

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
]


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

        from ..vs_proxy.vs_proxy import register_on_creation

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
        from ..vs_proxy.vs_proxy import core, register_on_destroy

        VSDebug._print_func(f"New core created with id: {core_id}")

        core.register_on_destroy(VSDebug._print_core_destroy, False)
        register_on_destroy(partial(VSDebug._print_destroy, core.env.env_id, core_id))

    @staticmethod
    def _print_destroy(env_id: int, core_id: int) -> None:
        VSDebug._print_func(f"Environment destroyed with id: {env_id}, current core id: {core_id}")

    @staticmethod
    def _print_core_destroy(_: int, core_id: int) -> None:
        VSDebug._print_func(f"Core destroyed with id: {core_id}")
