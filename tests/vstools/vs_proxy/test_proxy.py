from __future__ import annotations

import gc
import importlib
import inspect
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest
from jetpytools import CustomRuntimeError, CustomValueError
from vsengine.policy import ManagedEnvironment

import __main__
import vstools.vs_proxy.proxy as vs_proxy
from vstools import VSCoreProxy, core, vs
from vstools.vs_proxy.proxy import CoreProxy, FunctionProxy, PluginProxy, _find_ref

_creation_called_cores = list[int]()


def _on_creation_callback(core_id: int) -> None:
    _creation_called_cores.append(core_id)


def test_interactive_env_emulation(monkeypatch: pytest.MonkeyPatch) -> None:
    # Back up
    if hasattr(__main__, "__file__"):
        monkeypatch.delattr(__main__, "__file__")
    if "__vapoursynth__" in sys.modules:
        monkeypatch.delitem(sys.modules, "__vapoursynth__")

    # Reload proxy module to trigger initialization emulation
    importlib.reload(vs_proxy)

    assert "__vapoursynth__" in sys.modules
    assert hasattr(__main__, "__file__")

    # Test case where inspect.currentframe() is None to hit L787
    monkeypatch.delitem(sys.modules, "__vapoursynth__")
    monkeypatch.delattr(__main__, "__file__")
    monkeypatch.setattr(inspect, "currentframe", lambda: None)
    importlib.reload(vs_proxy)
    assert "__vapoursynth__" in sys.modules
    assert not hasattr(__main__, "__file__")

    # Test L801 (entry_filename resolves to a file whose parent is not in sys.path)
    monkeypatch.delitem(sys.modules, "__vapoursynth__")

    class MockFrame:
        def __init__(self) -> None:
            self.f_back = None
            self.f_code = type("Code", (), {"co_filename": "nonexistent_dir/dummy.py"})()

    monkeypatch.setattr(inspect, "currentframe", lambda: MockFrame())
    monkeypatch.setattr(Path, "exists", lambda self: True)

    orig_path = list(sys.path)
    try:
        importlib.reload(vs_proxy)
        assert any("nonexistent_dir" in p for p in sys.path)
    finally:
        sys.path = orig_path


@pytest.mark.vpy("no-core", "unique-core")
def test_callbacks_on_creation(vpy_stage: str, vpy_env_factory: Callable[[], ManagedEnvironment]) -> None:
    global _creation_called_cores
    _creation_called_cores.clear()

    if vpy_stage == "no-core":
        # Under no-core, core is not active
        assert not core.active

        immediate_run = vs.register_on_creation(_on_creation_callback)
        assert not immediate_run
        assert len(_creation_called_cores) == 0

        with vpy_env_factory().use():
            active_core = core.core
            assert len(_creation_called_cores) == 1
            assert _creation_called_cores[0] == id(active_core)

        assert vs.unregister_on_creation(_on_creation_callback)

    elif vpy_stage == "unique-core":
        assert core.active

        immediate_run = vs.register_on_creation(_on_creation_callback)
        assert immediate_run
        assert len(_creation_called_cores) == 1
        assert _creation_called_cores[0] == core.core_id

        assert vs.unregister_on_creation(_on_creation_callback)

    assert not vs.unregister_on_creation(_on_creation_callback)


@pytest.mark.vpy("no-core", "unique-core")
def test_environment_detection(vpy_stage: str) -> None:
    if vpy_stage == "no-core":
        assert not vs.has_environment()
        assert not core.active
        with pytest.raises(CustomRuntimeError, match=r"No environment is currently activated\."):
            vs.get_policy()
        with pytest.raises(CustomRuntimeError, match=r"Core hasn't been fetched yet!"):
            core.core_id
        with pytest.raises(RuntimeError, match=r"We are not running inside an environment\."):
            core.env.data

    elif vpy_stage == "unique-core":
        assert vs.has_environment()
        assert vs.get_policy()
        assert vs.get_policy_api()


def test_proxy_get_policy_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    # get_policy when environment data is None
    mock_env = MagicMock()
    mock_env.env.return_value = None
    with monkeypatch.context() as m:
        m.setattr(vs_proxy, "get_current_environment", lambda: mock_env)
        with pytest.raises(CustomRuntimeError, match=r"No environment is currently activated\."):
            vs_proxy.get_policy()
        with pytest.raises(CustomRuntimeError, match=r"No environment is currently activated\."):
            core.env.data

    # get_policy when policy is None (not found in gc)
    mock_env = MagicMock()
    mock_env.env.return_value = object()
    with monkeypatch.context() as m:
        m.setattr(vs_proxy, "get_current_environment", lambda: mock_env)
        m.setattr(vs_proxy, "_find_ref", lambda *args, **kwargs: None)
        with pytest.raises(CustomRuntimeError, match=r"No policy is currently registered\."):
            vs_proxy.get_policy()

    # get_policy_api when policy API is None
    with monkeypatch.context() as m:
        m.setattr(vs_proxy, "get_policy", lambda: object())
        m.setattr(vs_proxy, "_find_ref", lambda *args, **kwargs: None)
        with pytest.raises(CustomRuntimeError, match=r"No policy API found\."):
            vs_proxy.get_policy_api()


def test_find_ref_recursion() -> None:
    class DummyTarget: ...

    class DummyStart:
        def __init__(self, target: DummyTarget) -> None:
            self.ref_list = [target]

    target = DummyTarget()
    start = DummyStart(target)

    assert _find_ref(start, (DummyTarget,)) is target
    assert _find_ref(start, (DummyTarget,), it=0) is None


@pytest.mark.vpy("unique-core")
def test_core_proxy_proxied() -> None:
    assert core.active
    proxied_text = core.proxied.text
    assert proxied_text


@pytest.mark.vpy("unique-core")
def test_plugin_and_function_proxies() -> None:
    plugin = core.lazy.std
    assert isinstance(plugin, PluginProxy)

    func = plugin.BlankClip
    assert isinstance(func, FunctionProxy)
    assert func()


@pytest.mark.vpy("no-core", "unique-core")
def test_core_proxy_lazy(vpy_stage: str) -> None:
    if vpy_stage == "no-core":
        assert not core.active
        lazy_std = core.lazy.std
        assert isinstance(lazy_std, PluginProxy)
        assert not core.active

        lazy_func = lazy_std.BlankClip
        assert isinstance(lazy_func, FunctionProxy)
        assert not core.active

        with pytest.raises(vs.Error):
            lazy_func()

    elif vpy_stage == "unique-core":
        assert core.active
        assert core.lazy.std.BlankClip()


@pytest.mark.vpy("unique-core")
def test_vs_core_proxy_properties() -> None:
    active_core = core.core
    assert core.active
    assert core.core_id == id(active_core)

    core.set_affinity(threads=1, max_cache=100)
    assert active_core.num_threads == 1
    assert active_core.max_cache_size == 100


def test_function_proxy_getattr() -> None:
    plugin = core.lazy.std
    func = plugin.BlankClip
    assert getattr(func, "name") == "BlankClip"


def test_plugin_proxy_getattr_non_lazy() -> None:
    non_lazy_core = CoreProxy(core.core, core, False)
    plugin = non_lazy_core.std

    assert plugin.namespace == "std"
    assert plugin.name == "VapourSynth Core Functions"

    # Access a function
    assert type(plugin.BlankClip).__name__ == "FunctionProxy"

    # Access an attribute that is not a function
    assert plugin.namespace == "std"


def test_core_proxy_getattr_non_lazy() -> None:
    non_lazy_core = CoreProxy(core.core, core, False)
    # Access a plugin
    std_plugin = non_lazy_core.std
    assert type(std_plugin).__name__ == "PluginProxy"

    # Access a non-plugin attribute from core
    assert isinstance(non_lazy_core.num_threads, int)


def test_core_proxy_vs_core_ref_fallback() -> None:
    class DummyCore: ...

    # _own_core is True (should raise error when core is freed)
    dummy1 = DummyCore()
    vs_proxy1 = VSCoreProxy(dummy1)  # type: ignore[arg-type]
    core_proxy1 = CoreProxy(dummy1, vs_proxy1, True)  # type: ignore[arg-type]

    del dummy1
    gc.collect()

    with pytest.raises(CustomRuntimeError, match="The VapourSynth core has been freed!"):
        core_proxy1._vs_core_ref

    # _own_core is False (should fall back to global active core)
    dummy2 = DummyCore()
    vs_proxy2 = VSCoreProxy()
    core_proxy2 = CoreProxy(dummy2, vs_proxy2, True)  # type: ignore[arg-type]

    del dummy2
    gc.collect()

    assert core_proxy2._vs_core_ref


def test_environment_proxy_setattr(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyEnv: ...

    dummy_env = DummyEnv()
    monkeypatch.setattr("vstools.vs_proxy.proxy.get_current_environment", lambda: dummy_env)
    setattr(core.env, "test_attr", "hello")
    assert getattr(dummy_env, "test_attr") == "hello"


def test_vs_core_proxy_setattr() -> None:
    orig_threads = core.core.num_threads
    try:
        core.num_threads = 2
        assert core.core.num_threads == 2
    finally:
        core.core.num_threads = orig_threads


def test_vs_core_proxy_no_policy_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("vstools.vs_proxy.proxy.has_policy", lambda: False)
    with pytest.raises(CustomRuntimeError, match="No policy has been registered!"):
        core.env


def test_vs_core_proxy_proxied_caching() -> None:
    temp_proxy = VSCoreProxy()
    assert temp_proxy.proxied
    assert temp_proxy.lazy


def test_vs_core_proxy_set_affinity_args() -> None:
    orig_threads = core.core.num_threads
    try:
        core.set_affinity(threads=None)
        core.set_affinity(threads=0.5)
        core.set_affinity(threads=(0, 2))

        with pytest.raises(CustomValueError, match="threads` should be greater than 0"):
            core.set_affinity(threads=-0.5)
    finally:
        core.core.num_threads = orig_threads


def test_vs_core_proxy_core_freed_error() -> None:
    class DummyCore: ...

    dummy = cast(vs.Core, DummyCore())
    temp_proxy = VSCoreProxy(dummy)
    assert temp_proxy._core is dummy
    del dummy
    gc.collect()

    with pytest.raises(CustomRuntimeError, match="The core the proxy made reference to was freed!"):
        temp_proxy._core
