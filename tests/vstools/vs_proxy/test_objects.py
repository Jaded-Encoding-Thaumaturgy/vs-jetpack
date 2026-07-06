from __future__ import annotations

import gc
from collections.abc import Iterator
from typing import Any

import pytest

from vstools import VSObject, core
from vstools.vs_proxy.objects import _iterative_check


def test_iterative_check_cases() -> None:
    # VapourSynth object inside list
    assert _iterative_check([core.core])

    # Dict with non-string keys
    assert not _iterative_check({123: "value"})
    assert _iterative_check({core.core: "value"})

    # Custom iterable
    class CustomIterable:
        def __init__(self, items: list[Any]) -> None:
            self.items = items

        def __iter__(self) -> Iterator[Any]:
            return iter(self.items)

    assert _iterative_check(CustomIterable([core.core]))


def test_safe_vs_object_del() -> None:
    # dict attribute cleanup
    class CustomVSObj(VSObject):
        def __init__(self) -> None:
            self.node = core.std.BlankClip()

    obj = CustomVSObj()
    assert hasattr(obj, "node")
    obj.__vs_del__(0)
    assert not hasattr(obj, "node")

    # slots attribute cleanup
    class CustomVSObjSlots(VSObject):
        __slots__ = ("node", "other_val")

        def __init__(self) -> None:
            self.node = core.core.std.BlankClip()
            self.other_val = 123

    obj_slots = CustomVSObjSlots()
    assert hasattr(obj_slots, "node")
    obj_slots.__vs_del__(0)
    assert not hasattr(obj_slots, "node")
    assert obj_slots.other_val == 123


def test_register_vs_del_dead_weakref(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(5, logger="vstools.vs_proxy.objects")

    cbs = list[Any]()
    monkeypatch.setattr("vstools.vs_proxy.objects.register_on_creation", lambda cb: None)
    monkeypatch.setattr("vstools.vs_proxy.objects.register_on_destroy", cbs.append)

    class TempVSObj(VSObject): ...

    temp_obj = TempVSObj()
    reg_cb = getattr(temp_obj, "__vsdel_register")
    reg_cb(0)

    assert len(cbs) == 1
    destroy_cb = cbs[0]

    # Explicitly delete references to allow garbage collection
    del temp_obj
    del reg_cb
    gc.collect()

    # Calling destroy_cb should exit cleanly without raising an error because the weakref is dead
    destroy_cb()

    assert any("Dead object, skipping cleanup." in record.message for record in caplog.records)


def test_register_vs_del_success_logging(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(5, logger="vstools.vs_proxy.objects")

    cbs = list[Any]()
    monkeypatch.setattr("vstools.vs_proxy.objects.register_on_creation", lambda cb: None)
    monkeypatch.setattr("vstools.vs_proxy.objects.register_on_destroy", cbs.append)

    # Standard dunder
    class LoggingVSObj(VSObject): ...

    obj = LoggingVSObj()
    reg_cb = getattr(obj, "__vsdel_register")
    reg_cb(0)

    assert len(cbs) == 1
    destroy_cb = cbs[0]
    destroy_cb()

    assert any(
        "LoggingVSObj" in record.message and "has been freed using" in record.message for record in caplog.records
    )

    # Custom dunder
    class CustomDunderVSObj(VSObject):
        def __vs_del__(self, core_id: int) -> None:
            pass

    caplog.clear()
    obj_custom = CustomDunderVSObj()
    reg_cb_custom = getattr(obj_custom, "__vsdel_register")
    reg_cb_custom(0)

    assert len(cbs) == 2
    destroy_cb_custom = cbs[1]
    destroy_cb_custom()

    assert any(
        "CustomDunderVSObj" in record.message and "Custom dunder detected!" in record.message
        for record in caplog.records
    )


def test_vsobject_meta_slots_generation() -> None:
    class SlotsBase(VSObject):
        __slots__ = ("base_slot",)

    class SlotsChild(SlotsBase):
        __slots__ = ("child_slot",)

    class SlotsWithWeakref(VSObject):
        __slots__ = ("__weakref__", "some_slot")

    assert "base_slot" in SlotsBase.__slots__
    assert "child_slot" in SlotsChild.__slots__
    assert "some_slot" in SlotsWithWeakref.__slots__


def test_vsobject_new_type_error() -> None:
    class ObjWithArgs(VSObject):
        def __init__(self, a: int, b: int) -> None:
            self.a = a
            self.b = b

    # Instantiating with args triggers TypeError in super().__new__ and falls back
    obj = ObjWithArgs(1, 2)
    assert obj.a == 1
    assert obj.b == 2
