from __future__ import annotations

from abc import ABC, ABCMeta
from enum import Flag
from functools import partial
from itertools import chain
from logging import getLogger
from typing import TYPE_CHECKING, Any, Callable, Mapping, MutableMapping, MutableSequence, MutableSet, Self

from jetpytools import Singleton, classproperty

from vsjetpack import is_from_vs_module

from .proxy import core, register_on_creation, register_on_destroy

__all__ = ["VSDebug", "VSObject", "VSObjectABC", "VSObjectABCMeta", "VSObjectMeta", "vs_object"]


log = getLogger(__name__)


def _get_mangle_name(name: str) -> str:
    return "_" + name.lstrip("_")


def _iterative_check(x: Any) -> bool:
    stack = [x]

    while stack:
        current = stack.pop()

        if res := is_from_vs_module(current):
            return res

        if isinstance(current, (str, bytes, bytearray, Flag)):
            continue

        if isinstance(current, Mapping):
            for k, v in current.items():
                if isinstance(k, str):
                    if k.startswith("__"):
                        continue
                else:
                    stack.append(k)
                stack.append(v)
            continue

        if isinstance(current, (list, tuple, set, frozenset)):
            stack.extend(current)
            continue

        try:
            iterator = iter(current)
        except TypeError:
            continue
        else:
            stack.extend(iterator)

    return False


def _safe_vs_object_del(obj: Any) -> None:
    obj_dict = getattr(obj, "__dict__", None)
    if obj_dict is not None:
        for k, v in list(obj_dict.items()):
            if not k.startswith("__") and _iterative_check(v):
                delattr(obj, k)

    obj_slots = getattr(obj, "__slots__", None)
    # We only want to check the instances.
    if obj_slots is not None and isinstance(obj, VSObject):
        mname = _get_mangle_name(obj.__class__.__name__)

        for k in obj.__all_slots__:
            if k.startswith(("__", mname)):
                continue

            v = getattr(obj, k, None)

            if _iterative_check(v):
                delattr(obj, k)

    if isinstance(obj, (MutableMapping, MutableSequence, MutableSet)):
        obj.clear()


_clsregisters = "__clsvsdel_partial_register", "__clsvsdel_register"
_objregisters = "__vsdel_partial_register", "__vsdel_register"


def _register_vs_del(obj: VSObject | VSObjectMeta) -> None:
    """
    Register cleanup for both VSObject (instance-level) and VSObjectMeta (class-level).
    """
    prefix = ""

    if isinstance(obj, VSObjectMeta):
        del_method = "__cls_vs_del__"
        partial_attr, register_attr = _clsregisters
    else:
        del_method = "__vs_del__"
        partial_attr, register_attr = _objregisters

        if not hasattr(obj, "__dict__"):
            prefix = _get_mangle_name(obj.__class__.__name__)

    def del_register(core_id: int) -> None:
        def vsdel_partial_register() -> None:
            getattr(obj, del_method)(core_id)
            log.log(
                5,
                "%r has been freed using %r%s",
                getattr(obj, "__name__", obj.__class__.__name__),
                del_method,
                "... Custom dunder detected!"
                if (
                    (isinstance(obj, VSObject) and obj.__class__.__vs_del__ != VSObject.__vs_del__)
                    or (isinstance(obj, VSObjectMeta) and obj.__class__.__cls_vs_del__ != VSObjectMeta.__cls_vs_del__)
                )
                else "",
            )

        setattr(obj, prefix + partial_attr, vsdel_partial_register)
        core.register_on_destroy(vsdel_partial_register)

    setattr(obj, prefix + register_attr, del_register)
    register_on_creation(del_register)


class VSObjectMeta(type):
    """
    Metaclass for VSObject that ensures VapourSynth object lifecycle hooks are registered.

    This metaclass automatically registers the `__vs_del__` cleanup hook for any class that inherits from `VSObject`.
    By default, the cleanup mechanism will attempt to safely release any VapourSynth-bound objects
    when the core is freed, preventing resource leaks.
    """

    def __new__[MetaSelf: VSObjectMeta](
        mcls: type[MetaSelf], name: str, bases: tuple[type, ...], namespace: dict[str, Any], /, **kwargs: Any
    ) -> MetaSelf:
        """
        Extend classes with lifecycle cleanup hooks if they declare __slots__.

        If the class or any of its bases use __slots__, automatically injects the necessary attributes
        for VapourSynth cleanup registration.
        """
        if "__slots__" in namespace and (
            namespace["__slots__"] or any(getattr(base, "__slots__", ()) for base in bases)
        ):
            mname = _get_mangle_name(name)

            original_slots = tuple(namespace.get("__slots__", ()))
            extra_slots = tuple(f"{mname}{slot}" for slot in _objregisters)
            namespace["__slots__"] = (*extra_slots, *original_slots)

            for reg in _clsregisters:
                namespace[f"{mname}{reg}"] = None

        cls = super().__new__(mcls, name, bases, namespace, **kwargs)

        _register_vs_del(cls)

        log.log(5, "%s: Registering %r as VS object meta", mcls, cls)

        return cls

    def __cls_vs_del__(cls, core_id: int) -> None:
        _safe_vs_object_del(cls)


class VSObject(metaclass=VSObjectMeta):
    """
    Base class for objects bound to the lifecycle of a VapourSynth core.

    Subclasses can define the special dunder method `__vs_del__`, which is invoked when the VapourSynth core
    that owns the object is released.

    By default, this method will attempt to safely delete any references to VapourSynth objects held by the instance.
    Overriding `__vs_del__` is only necessary if you need custom cleanup logic beyond the default safe release.
    """

    __slots__ = ()

    if not TYPE_CHECKING:

        def __new__(cls, *args: Any, **kwargs: Any) -> Self:
            try:
                obj = super().__new__(cls, *args, **kwargs)
            except TypeError:
                obj = super().__new__(cls)

            log.log(5, "%s: Registering %r as VS object", "VSObject", cls)

            _register_vs_del(obj)
            return obj

    @classproperty.cached
    @classmethod
    def __all_slots__(cls) -> tuple[str, ...]:
        slots = dict.fromkeys(chain.from_iterable(getattr(base, "__slots__", ()) for base in reversed(cls.mro())))
        return tuple(slots)

    def __vs_del__(self, core_id: int) -> None:
        _safe_vs_object_del(self)


class VSObjectABCMeta(VSObjectMeta, ABCMeta):
    """
    Metaclass for abstract VSObject classes.
    """


class VSObjectABC(VSObject, ABC, metaclass=VSObjectABCMeta):
    """
    Abstract base class for VSObject subclasses.
    """

    __slots__ = ()


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
        VSDebug._print_func(f"New core created with id: {core_id}")

        core.register_on_destroy(VSDebug._print_core_destroy, False)
        register_on_destroy(partial(VSDebug._print_destroy, core.env.env_id, core_id))

    @staticmethod
    def _print_destroy(env_id: int, core_id: int) -> None:
        VSDebug._print_func(f"Environment destroyed with id: {env_id}, current core id: {core_id}")

    @staticmethod
    def _print_core_destroy(_: int, core_id: int) -> None:
        VSDebug._print_func(f"Core destroyed with id: {core_id}")


vs_object = VSObject
"""Deprecated alias for VSObject"""
