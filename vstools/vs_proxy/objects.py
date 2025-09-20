from __future__ import annotations

from abc import ABC, ABCMeta
from contextlib import suppress
from enum import Flag
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Mapping,
    MutableMapping,
    MutableSequence,
    MutableSet,
)

from typing_extensions import Self

from .proxy import core, register_on_creation

__all__ = ["VSObject", "VSObjectABC", "VSObjectABCMeta", "VSObjectMeta", "vs_object"]


def _iterative_check(x: Any) -> bool:
    stack = [x]

    while stack:
        current = stack.pop()

        if getattr(current, "__module__", "") == "vapoursynth":
            return True

        if isinstance(current, (str, bytes, bytearray, Flag)):
            continue

        if isinstance(current, Mapping):
            for k, v in current.items():
                if not (isinstance(k, str) and k.startswith("__")):
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
    # Handle normal attributes
    obj_dict = getattr(obj, "__dict__", None)
    if obj_dict is not None:
        for k, v in list(obj_dict.items()):
            if not k.startswith("__") and _iterative_check(v):
                with suppress(AttributeError):
                    delattr(obj, k)

    # Handle slots
    obj_slots = getattr(obj, "__slots__", None)
    if obj_slots:
        for k in obj_slots:
            if k.startswith("__"):
                continue
            v = getattr(obj, k, None)
            if _iterative_check(v):
                with suppress(AttributeError):
                    delattr(obj, k)

    # Handle containers
    if isinstance(obj, (MutableMapping, MutableSequence, MutableSet)):
        obj.clear()


def _register__vs_del__(obj: VSObject | VSObjectMeta) -> None:
    def _register(core_id: int) -> None:
        def __vsdel_register(core_id: int) -> None:
            if isinstance(obj, VSObject):
                obj.__vs_del__(core_id)
            else:
                obj.__cls_vs_del__(core_id)

        vsdel_partial_register = partial(__vsdel_register, core_id)
        setattr(obj, "__vsdel_partial_register", vsdel_partial_register)

        core.register_on_destroy(vsdel_partial_register)

    # [un]register_on_creation/destroy will only hold a weakref to the object
    setattr(obj, "__vsdel_register", _register)
    register_on_creation(_register)


class VSObjectMeta(type):
    def __new__(mcls, name: str, bases: tuple[type, ...], namespace: dict[str, Any], /, **kwargs: Any) -> VSObjectMeta:
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)

        _register__vs_del__(cls)
        return cls

    def __cls_vs_del__(cls, core_id: int) -> None:
        _safe_vs_object_del(cls)


class VSObject(metaclass=VSObjectMeta):
    """
    Special object that follows the lifecycle of the VapourSynth environment/core.

    If a special dunder is created, __vs_del__, it will be called when the environment is getting deleted.

    This is especially useful if you have to hold a reference to a VideoNode or Plugin/Function object
    in this object as you need to remove it for the VapourSynth core to be freed correctly.
    """

    __slots__ = ()

    if not TYPE_CHECKING:

        def __new__(cls, *args: Any, **kwargs: Any) -> Self:
            try:
                obj = super().__new__(cls, *args, **kwargs)
            except TypeError:
                obj = super().__new__(cls)

            _register__vs_del__(obj)
            return obj

    def __vs_del__(self, core_id: int) -> None:
        """
        Special dunder that will be called when a core is getting freed.
        """
        _safe_vs_object_del(self)


vs_object = VSObject


class VSObjectABCMeta(VSObjectMeta, ABCMeta): ...


class VSObjectABC(VSObject, ABC, metaclass=VSObjectABCMeta):
    __slots__ = ()
