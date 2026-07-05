from __future__ import annotations

import gc
import sys
import weakref
from collections.abc import Callable, Iterable
from itertools import chain
from operator import attrgetter
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal

from jetpytools import CustomRuntimeError, CustomValueError
from vapoursynth import (
    AUDIO,
    BACK_CENTER,
    BACK_LEFT,
    BACK_RIGHT,
    CHROMA_BOTTOM,
    CHROMA_BOTTOM_LEFT,
    CHROMA_CENTER,
    CHROMA_LEFT,
    CHROMA_TOP,
    CHROMA_TOP_LEFT,
    DISABLE_AUTO_LOADING,
    DISABLE_LIBRARY_UNLOADING,
    ENABLE_GRAPH_INSPECTION,
    FIELD_BOTTOM,
    FIELD_PROGRESSIVE,
    FIELD_TOP,
    FLOAT,
    FRAME_STATE,
    FRONT_CENTER,
    FRONT_LEFT,
    FRONT_LEFT_OF_CENTER,
    FRONT_RIGHT,
    FRONT_RIGHT_OF_CENTER,
    GRAY,
    INTEGER,
    LOW_FREQUENCY,
    LOW_FREQUENCY2,
    MATRIX_BT470_BG,
    MATRIX_BT709,
    MATRIX_BT2020_CL,
    MATRIX_BT2020_NCL,
    MATRIX_CHROMATICITY_DERIVED_CL,
    MATRIX_CHROMATICITY_DERIVED_NCL,
    MATRIX_FCC,
    MATRIX_ICTCP,
    MATRIX_RGB,
    MATRIX_ST170_M,
    MATRIX_UNSPECIFIED,
    MATRIX_YCGCO,
    MESSAGE_TYPE_CRITICAL,
    MESSAGE_TYPE_DEBUG,
    MESSAGE_TYPE_FATAL,
    MESSAGE_TYPE_INFORMATION,
    MESSAGE_TYPE_WARNING,
    NONE,
    PARALLEL,
    PARALLEL_REQUESTS,
    PRIMARIES_BT470_BG,
    PRIMARIES_BT470_M,
    PRIMARIES_BT709,
    PRIMARIES_BT2020,
    PRIMARIES_EBU3213_E,
    PRIMARIES_FILM,
    PRIMARIES_ST170_M,
    PRIMARIES_ST240_M,
    PRIMARIES_ST428,
    PRIMARIES_ST431_2,
    PRIMARIES_ST432_1,
    PRIMARIES_UNSPECIFIED,
    RANGE_FULL,
    RANGE_LIMITED,
    RGB,
    SIDE_LEFT,
    SIDE_RIGHT,
    STEREO_LEFT,
    STEREO_RIGHT,
    SURROUND_DIRECT_LEFT,
    SURROUND_DIRECT_RIGHT,
    TOP_BACK_CENTER,
    TOP_BACK_LEFT,
    TOP_BACK_RIGHT,
    TOP_CENTER,
    TOP_FRONT_CENTER,
    TOP_FRONT_LEFT,
    TOP_FRONT_RIGHT,
    TRANSFER_ARIB_B67,
    TRANSFER_BT470_BG,
    TRANSFER_BT470_M,
    TRANSFER_BT601,
    TRANSFER_BT709,
    TRANSFER_BT2020_10,
    TRANSFER_BT2020_12,
    TRANSFER_IEC_61966_2_1,
    TRANSFER_IEC_61966_2_4,
    TRANSFER_LINEAR,
    TRANSFER_LOG_100,
    TRANSFER_LOG_316,
    TRANSFER_ST240_M,
    TRANSFER_ST428,
    TRANSFER_ST2084,
    TRANSFER_UNSPECIFIED,
    UNDEFINED,
    UNORDERED,
    VIDEO,
    WIDE_LEFT,
    WIDE_RIGHT,
    YUV,
    AudioChannels,
    AudioFrame,
    AudioNode,
    ChromaLocation,
    ColorFamily,
    ColorPrimaries,
    ColorRange,
    Core,
    CoreCreationFlags,
    Environment,
    EnvironmentData,
    EnvironmentPolicy,
    EnvironmentPolicyAPI,
    Error,
    FieldBased,
    FilterMode,
    FrameProps,
    FramePtr,
    Func,
    FuncData,
    Function,
    LogHandle,
    MatrixCoefficients,
    MediaType,
    MessageType,
    Plugin,
    Range,
    RawFrame,
    RawNode,
    SampleType,
    StandaloneEnvironmentPolicy,
    TransferCharacteristics,
    VideoFormat,
    VideoFrame,
    VideoNode,
    VideoOutputTuple,
    VSScriptEnvironmentPolicy,
    __api_version__,
    __version__,
    _CoreProxy,
    _FastManager,
    clear_output,
    clear_outputs,
    construct_signature,
    get_current_environment,
    get_output,
    get_outputs,
    has_policy,
    register_on_destroy,
    register_policy,
    unregister_on_destroy,
)
from vapoursynth import __file__ as vs_file
from vapoursynth import __pyx_capi__ as pyx_capi  # type: ignore[attr-defined]
from vapoursynth import _construct_parameter as construct_parameter
from vapoursynth import _construct_type as construct_type
from vapoursynth import _try_enable_introspection as try_enable_introspection

import __main__
from vsjetpack import TypeForm, deprecated

from .enums import (
    GRAY8,
    GRAY9,
    GRAY10,
    GRAY11,
    GRAY12,
    GRAY13,
    GRAY14,
    GRAY15,
    GRAY16,
    GRAY17,
    GRAY18,
    GRAY19,
    GRAY20,
    GRAY21,
    GRAY22,
    GRAY23,
    GRAY24,
    GRAY25,
    GRAY26,
    GRAY27,
    GRAY28,
    GRAY29,
    GRAY30,
    GRAY31,
    GRAY32,
    GRAYH,
    GRAYS,
    RGB24,
    RGB27,
    RGB30,
    RGB33,
    RGB36,
    RGB39,
    RGB42,
    RGB45,
    RGB48,
    RGB51,
    RGB54,
    RGB57,
    RGB60,
    RGB63,
    RGB66,
    RGB69,
    RGB72,
    RGB75,
    RGB78,
    RGB81,
    RGB84,
    RGB87,
    RGB90,
    RGB93,
    RGB96,
    RGBH,
    RGBS,
    YUV410P8,
    YUV410P9,
    YUV410P10,
    YUV410P11,
    YUV410P12,
    YUV410P13,
    YUV410P14,
    YUV410P15,
    YUV410P16,
    YUV410P17,
    YUV410P18,
    YUV410P19,
    YUV410P20,
    YUV410P21,
    YUV410P22,
    YUV410P23,
    YUV410P24,
    YUV410P25,
    YUV410P26,
    YUV410P27,
    YUV410P28,
    YUV410P29,
    YUV410P30,
    YUV410P31,
    YUV410P32,
    YUV410PH,
    YUV410PS,
    YUV411P8,
    YUV411P9,
    YUV411P10,
    YUV411P11,
    YUV411P12,
    YUV411P13,
    YUV411P14,
    YUV411P15,
    YUV411P16,
    YUV411P17,
    YUV411P18,
    YUV411P19,
    YUV411P20,
    YUV411P21,
    YUV411P22,
    YUV411P23,
    YUV411P24,
    YUV411P25,
    YUV411P26,
    YUV411P27,
    YUV411P28,
    YUV411P29,
    YUV411P30,
    YUV411P31,
    YUV411P32,
    YUV411PH,
    YUV411PS,
    YUV420P8,
    YUV420P9,
    YUV420P10,
    YUV420P11,
    YUV420P12,
    YUV420P13,
    YUV420P14,
    YUV420P15,
    YUV420P16,
    YUV420P17,
    YUV420P18,
    YUV420P19,
    YUV420P20,
    YUV420P21,
    YUV420P22,
    YUV420P23,
    YUV420P24,
    YUV420P25,
    YUV420P26,
    YUV420P27,
    YUV420P28,
    YUV420P29,
    YUV420P30,
    YUV420P31,
    YUV420P32,
    YUV420PH,
    YUV420PS,
    YUV422P8,
    YUV422P9,
    YUV422P10,
    YUV422P11,
    YUV422P12,
    YUV422P13,
    YUV422P14,
    YUV422P15,
    YUV422P16,
    YUV422P17,
    YUV422P18,
    YUV422P19,
    YUV422P20,
    YUV422P21,
    YUV422P22,
    YUV422P23,
    YUV422P24,
    YUV422P25,
    YUV422P26,
    YUV422P27,
    YUV422P28,
    YUV422P29,
    YUV422P30,
    YUV422P31,
    YUV422P32,
    YUV422PH,
    YUV422PS,
    YUV440P8,
    YUV440P9,
    YUV440P10,
    YUV440P11,
    YUV440P12,
    YUV440P13,
    YUV440P14,
    YUV440P15,
    YUV440P16,
    YUV440P17,
    YUV440P18,
    YUV440P19,
    YUV440P20,
    YUV440P21,
    YUV440P22,
    YUV440P23,
    YUV440P24,
    YUV440P25,
    YUV440P26,
    YUV440P27,
    YUV440P28,
    YUV440P29,
    YUV440P30,
    YUV440P31,
    YUV440P32,
    YUV440PH,
    YUV440PS,
    YUV444P8,
    YUV444P9,
    YUV444P10,
    YUV444P11,
    YUV444P12,
    YUV444P13,
    YUV444P14,
    YUV444P15,
    YUV444P16,
    YUV444P17,
    YUV444P18,
    YUV444P19,
    YUV444P20,
    YUV444P21,
    YUV444P22,
    YUV444P23,
    YUV444P24,
    YUV444P25,
    YUV444P26,
    YUV444P27,
    YUV444P28,
    YUV444P29,
    YUV444P30,
    YUV444P31,
    YUV444P32,
    YUV444PH,
    YUV444PS,
    PresetVideoFormat,
)

__all__ = [
    "AUDIO",
    "BACK_CENTER",
    "BACK_LEFT",
    "BACK_RIGHT",
    "CHROMA_BOTTOM",
    "CHROMA_BOTTOM_LEFT",
    "CHROMA_CENTER",
    "CHROMA_LEFT",
    "CHROMA_TOP",
    "CHROMA_TOP_LEFT",
    "DISABLE_AUTO_LOADING",
    "DISABLE_LIBRARY_UNLOADING",
    "ENABLE_GRAPH_INSPECTION",
    "FIELD_BOTTOM",
    "FIELD_PROGRESSIVE",
    "FIELD_TOP",
    "FLOAT",
    "FRAME_STATE",
    "FRONT_CENTER",
    "FRONT_LEFT",
    "FRONT_LEFT_OF_CENTER",
    "FRONT_RIGHT",
    "FRONT_RIGHT_OF_CENTER",
    "GRAY",
    "GRAY8",
    "GRAY9",
    "GRAY10",
    "GRAY11",
    "GRAY12",
    "GRAY13",
    "GRAY14",
    "GRAY15",
    "GRAY16",
    "GRAY17",
    "GRAY18",
    "GRAY19",
    "GRAY20",
    "GRAY21",
    "GRAY22",
    "GRAY23",
    "GRAY24",
    "GRAY25",
    "GRAY26",
    "GRAY27",
    "GRAY28",
    "GRAY29",
    "GRAY30",
    "GRAY31",
    "GRAY32",
    "GRAYH",
    "GRAYS",
    "INTEGER",
    "LOW_FREQUENCY",
    "LOW_FREQUENCY2",
    "MATRIX_BT470_BG",
    "MATRIX_BT709",
    "MATRIX_BT2020_CL",
    "MATRIX_BT2020_NCL",
    "MATRIX_CHROMATICITY_DERIVED_CL",
    "MATRIX_CHROMATICITY_DERIVED_NCL",
    "MATRIX_FCC",
    "MATRIX_ICTCP",
    "MATRIX_RGB",
    "MATRIX_ST170_M",
    "MATRIX_UNSPECIFIED",
    "MATRIX_YCGCO",
    "MESSAGE_TYPE_CRITICAL",
    "MESSAGE_TYPE_DEBUG",
    "MESSAGE_TYPE_FATAL",
    "MESSAGE_TYPE_INFORMATION",
    "MESSAGE_TYPE_WARNING",
    "NONE",
    "PARALLEL",
    "PARALLEL_REQUESTS",
    "PRIMARIES_BT470_BG",
    "PRIMARIES_BT470_M",
    "PRIMARIES_BT709",
    "PRIMARIES_BT2020",
    "PRIMARIES_EBU3213_E",
    "PRIMARIES_FILM",
    "PRIMARIES_ST170_M",
    "PRIMARIES_ST240_M",
    "PRIMARIES_ST428",
    "PRIMARIES_ST431_2",
    "PRIMARIES_ST432_1",
    "PRIMARIES_UNSPECIFIED",
    "RANGE_FULL",
    "RANGE_LIMITED",
    "RGB",
    "RGB24",
    "RGB27",
    "RGB30",
    "RGB33",
    "RGB36",
    "RGB39",
    "RGB42",
    "RGB45",
    "RGB48",
    "RGB51",
    "RGB54",
    "RGB57",
    "RGB60",
    "RGB63",
    "RGB66",
    "RGB69",
    "RGB72",
    "RGB75",
    "RGB78",
    "RGB81",
    "RGB84",
    "RGB87",
    "RGB90",
    "RGB93",
    "RGB96",
    "RGBH",
    "RGBS",
    "SIDE_LEFT",
    "SIDE_RIGHT",
    "STEREO_LEFT",
    "STEREO_RIGHT",
    "SURROUND_DIRECT_LEFT",
    "SURROUND_DIRECT_RIGHT",
    "TOP_BACK_CENTER",
    "TOP_BACK_LEFT",
    "TOP_BACK_RIGHT",
    "TOP_CENTER",
    "TOP_FRONT_CENTER",
    "TOP_FRONT_LEFT",
    "TOP_FRONT_RIGHT",
    "TRANSFER_ARIB_B67",
    "TRANSFER_BT470_BG",
    "TRANSFER_BT470_M",
    "TRANSFER_BT601",
    "TRANSFER_BT709",
    "TRANSFER_BT2020_10",
    "TRANSFER_BT2020_12",
    "TRANSFER_IEC_61966_2_1",
    "TRANSFER_IEC_61966_2_4",
    "TRANSFER_LINEAR",
    "TRANSFER_LOG_100",
    "TRANSFER_LOG_316",
    "TRANSFER_ST240_M",
    "TRANSFER_ST428",
    "TRANSFER_ST2084",
    "TRANSFER_UNSPECIFIED",
    "UNDEFINED",
    "UNORDERED",
    "VIDEO",
    "WIDE_LEFT",
    "WIDE_RIGHT",
    "YUV",
    "YUV410P8",
    "YUV410P9",
    "YUV410P10",
    "YUV410P11",
    "YUV410P12",
    "YUV410P13",
    "YUV410P14",
    "YUV410P15",
    "YUV410P16",
    "YUV410P17",
    "YUV410P18",
    "YUV410P19",
    "YUV410P20",
    "YUV410P21",
    "YUV410P22",
    "YUV410P23",
    "YUV410P24",
    "YUV410P25",
    "YUV410P26",
    "YUV410P27",
    "YUV410P28",
    "YUV410P29",
    "YUV410P30",
    "YUV410P31",
    "YUV410P32",
    "YUV410PH",
    "YUV410PS",
    "YUV411P8",
    "YUV411P9",
    "YUV411P10",
    "YUV411P11",
    "YUV411P12",
    "YUV411P13",
    "YUV411P14",
    "YUV411P15",
    "YUV411P16",
    "YUV411P17",
    "YUV411P18",
    "YUV411P19",
    "YUV411P20",
    "YUV411P21",
    "YUV411P22",
    "YUV411P23",
    "YUV411P24",
    "YUV411P25",
    "YUV411P26",
    "YUV411P27",
    "YUV411P28",
    "YUV411P29",
    "YUV411P30",
    "YUV411P31",
    "YUV411P32",
    "YUV411PH",
    "YUV411PS",
    "YUV420P8",
    "YUV420P9",
    "YUV420P10",
    "YUV420P11",
    "YUV420P12",
    "YUV420P13",
    "YUV420P14",
    "YUV420P15",
    "YUV420P16",
    "YUV420P17",
    "YUV420P18",
    "YUV420P19",
    "YUV420P20",
    "YUV420P21",
    "YUV420P22",
    "YUV420P23",
    "YUV420P24",
    "YUV420P25",
    "YUV420P26",
    "YUV420P27",
    "YUV420P28",
    "YUV420P29",
    "YUV420P30",
    "YUV420P31",
    "YUV420P32",
    "YUV420PH",
    "YUV420PS",
    "YUV422P8",
    "YUV422P9",
    "YUV422P10",
    "YUV422P11",
    "YUV422P12",
    "YUV422P13",
    "YUV422P14",
    "YUV422P15",
    "YUV422P16",
    "YUV422P17",
    "YUV422P18",
    "YUV422P19",
    "YUV422P20",
    "YUV422P21",
    "YUV422P22",
    "YUV422P23",
    "YUV422P24",
    "YUV422P25",
    "YUV422P26",
    "YUV422P27",
    "YUV422P28",
    "YUV422P29",
    "YUV422P30",
    "YUV422P31",
    "YUV422P32",
    "YUV422PH",
    "YUV422PS",
    "YUV440P8",
    "YUV440P9",
    "YUV440P10",
    "YUV440P11",
    "YUV440P12",
    "YUV440P13",
    "YUV440P14",
    "YUV440P15",
    "YUV440P16",
    "YUV440P17",
    "YUV440P18",
    "YUV440P19",
    "YUV440P20",
    "YUV440P21",
    "YUV440P22",
    "YUV440P23",
    "YUV440P24",
    "YUV440P25",
    "YUV440P26",
    "YUV440P27",
    "YUV440P28",
    "YUV440P29",
    "YUV440P30",
    "YUV440P31",
    "YUV440P32",
    "YUV440PH",
    "YUV440PS",
    "YUV444P8",
    "YUV444P9",
    "YUV444P10",
    "YUV444P11",
    "YUV444P12",
    "YUV444P13",
    "YUV444P14",
    "YUV444P15",
    "YUV444P16",
    "YUV444P17",
    "YUV444P18",
    "YUV444P19",
    "YUV444P20",
    "YUV444P21",
    "YUV444P22",
    "YUV444P23",
    "YUV444P24",
    "YUV444P25",
    "YUV444P26",
    "YUV444P27",
    "YUV444P28",
    "YUV444P29",
    "YUV444P30",
    "YUV444P31",
    "YUV444P32",
    "YUV444PH",
    "YUV444PS",
    "AudioChannels",
    "AudioFrame",
    "AudioNode",
    "ChromaLocation",
    "ColorFamily",
    "ColorPrimaries",
    "ColorRange",
    "Core",
    "CoreCreationFlags",
    "Environment",
    "EnvironmentData",
    "EnvironmentPolicy",
    "EnvironmentPolicyAPI",
    "Error",
    "FieldBased",
    "FilterMode",
    "FrameProps",
    "FramePtr",
    "Func",
    "FuncData",
    "Function",
    "LogHandle",
    "MatrixCoefficients",
    "MediaType",
    "MessageType",
    "Plugin",
    "PresetVideoFormat",
    "Range",
    "RawFrame",
    "RawNode",
    "SampleType",
    "StandaloneEnvironmentPolicy",
    "TransferCharacteristics",
    "VSScriptEnvironmentPolicy",
    "VideoFormat",
    "VideoFrame",
    "VideoNode",
    "VideoOutputTuple",
    "_CoreProxy",
    "__all__",
    "__api_version__",
    "__version__",
    "clear_cache",
    "clear_output",
    "clear_outputs",
    "construct_parameter",
    "construct_signature",
    "construct_type",
    "core",
    "get_current_environment",
    "get_output",
    "get_outputs",
    "get_policy",
    "get_policy_api",
    "has_environment",
    "has_policy",
    "pyx_capi",
    "register_on_creation",
    "register_on_destroy",
    "register_policy",
    "try_enable_introspection",
    "unregister_on_creation",
    "unregister_on_destroy",
    "vs_file",
]

# Compatibility helper designed to fake a standard script execution environment
# when running in interactive environments (such as Jupyter Notebooks, Python REPL, or IDE interactive windows)
# or embedded hosts where __main__.__file__ is not defined.
if not hasattr(__main__, "__file__") and "__vapoursynth__" not in sys.modules:
    import inspect

    # Walk up the call stack to find the entrypoint frame.
    frame = inspect.currentframe()
    try:
        if frame is not None:
            while frame.f_back:
                frame = frame.f_back
            entry_filename = frame.f_code.co_filename
        else:
            entry_filename = ""
    finally:
        del frame

    # Create a stub '__vapoursynth__' module to mimic the script execution context
    sys.modules["__vapoursynth__"] = ModuleType("__vapoursynth__")

    if entry_filename:
        # Resolve the absolute path to the entrypoint script
        script_path = Path(entry_filename).resolve()
        # Set the '__file__' attribute to emulate standard script execution
        sys.modules["__vapoursynth__"].__file__ = __main__.__file__ = str(script_path)
        # If the file exists on disk, add its directory to sys.path so local imports resolve correctly
        if script_path.exists() and (p := str(script_path.parent)) not in sys.path:
            sys.path.append(p)


def register_on_creation(callback: Callable[[int], None]) -> bool:
    """
    Register a callback on every core creation.

    Note:
        Callbacks are stored using weak references to prevent memory leaks from dynamic closures and bound methods.
        If you pass an inline lambda, transient bound method, or dynamically created function,
        it will be immediately garbage-collected and will not run.

        To prevent this:

        - Use module-level functions (which are strongly referenced by the module).
        - Keep a strong reference to the callback elsewhere (e.g. store it as an attribute on a persistent object).

    Returns whether the callback was executed immediately because a core is already active.
    """
    _core_on_creation_callbacks.add(callback)

    # If a core is already active, the catch-up logic is triggered immediately.
    # We trigger '_core_with_cb', which will execute any registered callbacks
    # that haven't been run for the current core instance yet.
    if has_policy() and has_environment() and core.active:
        with get_current_environment().use():
            core._core_with_cb
        return True

    return False


def unregister_on_creation(callback: Callable[[int], None]) -> bool:
    """
    Unregister this callback from every core creation.

    Returns whether the callback was successfully unregistered.
    """
    try:
        _core_on_creation_callbacks.remove(callback)
        return True
    except KeyError:
        return False


@deprecated("This function is deprecated. Use `core.clear_cache()` instead.", category=DeprecationWarning)
def clear_cache() -> None:
    try:
        cache_size = int(core.max_cache_size)
        core.max_cache_size = 1
        try:
            for output in get_outputs().values():
                if isinstance(output, VideoOutputTuple):
                    output.clip.get_frame(0).close()
                    break
        except Exception:
            core.std.BlankClip().get_frame(0).close()
        core.max_cache_size = cache_size
    except Exception:
        ...


def has_environment() -> bool:
    """Check if the current thread is running inside an environment."""
    try:
        return not not get_current_environment()  # noqa: SIM208
    except RuntimeError:
        return False


def get_policy() -> EnvironmentPolicy | VSScriptEnvironmentPolicy | StandaloneEnvironmentPolicy:
    """Retrieve the currently active VapourSynth EnvironmentPolicy."""
    if (data := get_current_environment().env()) is None:
        raise CustomRuntimeError("No environment is currently activated.")

    policy = _find_ref(data, (EnvironmentPolicy, VSScriptEnvironmentPolicy, StandaloneEnvironmentPolicy))

    if policy is None:
        raise CustomRuntimeError("No policy is currently registered.")

    return policy  # type: ignore[return-value]


def get_policy_api() -> EnvironmentPolicyAPI:
    """Retrieve the VapourSynth EnvironmentPolicyAPI bound to the currently registered policy."""
    api = _find_ref(get_policy(), (EnvironmentPolicyAPI,))

    if api is None:
        raise CustomRuntimeError("No policy API is currently registered.")

    return api


def _find_ref[T](
    start_data: Any,
    to_return: tuple[TypeForm[T], ...],
    it: int = 3,
    seen: set[int] | None = None,
) -> T | None:
    """
    Recursively search the garbage collector's referents and referrers
    to locate an active instance of specific types associated with the starting object.
    """
    if not it:
        return None

    if seen is None:
        seen = set()

    start_id = id(start_data)

    if start_id in seen:
        return None

    seen.add(start_id)

    for obj in chain(gc.get_referents(start_data), gc.get_referrers(start_data)):
        if isinstance(obj, to_return):  # type: ignore[arg-type]
            return obj

        obj_id = id(obj)

        if obj_id in seen:
            continue

        seen.add(obj_id)

        if isinstance(obj, dict) and "__name__" in obj:
            continue

        if isinstance(obj, (Core, _CoreProxy, CoreProxy, _FastManager)):
            continue

        for o in gc.get_referents(obj):
            if isinstance(o, to_return):  # type: ignore[arg-type]
                return o

            value = _find_ref(o, to_return, it - 1, seen)

            if value is not None:
                return value

    return None


if TYPE_CHECKING:

    class _FunctionProxyBase(Function): ...

    class _PluginProxyBase(Plugin): ...

    class _CoreProxyBase(_CoreProxy): ...

    class _EnvironmentProxyBase(Environment): ...
else:
    _FunctionProxyBase = _PluginProxyBase = _CoreProxyBase = _EnvironmentProxyBase = object


class FunctionProxy(_FunctionProxyBase):
    """
    A lazy proxy wrapper for a VapourSynth plugin function.

    Defers lookup and resolution of the underlying function until it is called or its attributes are accessed.
    """

    if not TYPE_CHECKING:
        __isabstractmethod__ = False

    def __init__(self, plugin: PluginProxy, func_name: str) -> None:
        self.__dict__["func_ref"] = (plugin, func_name)

    def __getattr__(self, name: str) -> Function:
        return getattr(self._vs_function, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._vs_function(*args, **kwargs)

    @property
    def _vs_function(self) -> Function:
        plugin, func_name = self._func_ref
        core, namespace = plugin._plugin_ref
        vs_core = core._vs_core_ref

        return attrgetter(f"{namespace}.{func_name}")(vs_core)

    @property
    def _func_ref(self) -> tuple[PluginProxy, str]:
        return self.__dict__["func_ref"]


class PluginProxy(_PluginProxyBase):
    """
    A lazy proxy wrapper for a VapourSynth plugin.

    Delegates attribute lookup to return a FunctionProxy
    or resolve the underlying VapourSynth Plugin's functions dynamically.
    """

    def __init__(self, core: CoreProxy, namespace: str) -> None:
        self.__dict__["plugin_ref"] = (core, namespace)

    def __getattr__(self, name: str) -> Function:
        core, namespace = self._plugin_ref

        if core.lazy and name not in Plugin.__dict__:
            return FunctionProxy(self, name)

        plugin: Plugin = getattr(core._vs_core_ref, namespace)

        if name in (f.name for f in plugin.functions()):
            return FunctionProxy(self, name)

        return getattr(plugin, name)

    @property
    def _plugin_ref(self) -> tuple[CoreProxy, str]:
        return self.__dict__["plugin_ref"]


class CoreProxy(_CoreProxyBase):
    """
    A lazy proxy wrapper for the VapourSynth Core.

    Supports deferred, lazy retrieval of plugins and functions
    to prevent premature core initialization and facilitate safe reference holding.
    """

    def __init__(self, core: Core | None, vs_proxy: VSCoreProxy, lazy: bool) -> None:
        self.lazy = lazy
        self.__dict__["vs_core_ref"] = (core and weakref.ref(core), vs_proxy)

    def __getattr__(self, name: str) -> Plugin:
        if self.lazy and name not in Core.__dict__:
            return PluginProxy(self, name)

        core = self._vs_core_ref

        if name in (p.namespace for p in core.plugins()):
            return PluginProxy(self, name)

        return getattr(core, name)

    @property
    def _vs_core_ref(self) -> Core:
        vs_core_ref, vs_proxy = self.__dict__["vs_core_ref"]

        vs_core = vs_core_ref and vs_core_ref()

        if vs_core_ref and vs_core is None:
            if object.__getattribute__(vs_proxy, "_own_core"):
                raise CustomRuntimeError("The VapourSynth core has been freed!", CoreProxy)

            vs_core = vs_proxy._core
            self.__dict__["vs_core_ref"] = (vs_core and weakref.ref(vs_core), vs_proxy)

        return vs_core or vs_proxy._core_with_cb


class EnvironmentProxy(_EnvironmentProxyBase):
    """A proxy wrapper around the active VapourSynth Environment."""

    def __getattr__(self, name: str) -> Plugin:
        return getattr(get_current_environment(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        return setattr(get_current_environment(), name, value)

    @property
    def data(self) -> EnvironmentData:
        """
        Retrieve the underlying EnvironmentData object.
        """
        if data := self.env():
            return data

        raise CustomRuntimeError("No environment is currently activated.")

    @property
    @deprecated("Accessing this attribute is deprecated. Use vs.get_policy() instead.", category=DeprecationWarning)
    def policy(self) -> EnvironmentPolicy | VSScriptEnvironmentPolicy | StandaloneEnvironmentPolicy:
        return get_policy()

    @property
    @deprecated("Accessing this attribute is deprecated. Use vs.get_policy_api() instead.", category=DeprecationWarning)
    def api(self) -> EnvironmentPolicyAPI:
        return get_policy_api()

    @property
    def has_core(self) -> bool:
        """
        Check if the active EnvironmentData has an instantiated VapourSynth Core.

        This avoids triggering the lazy creation of the Core.
        """
        return any(isinstance(ref, (Core, CoreProxy)) for ref in gc.get_referents(self.data))


class VSCoreProxy(_CoreProxyBase):
    """
    Class for wrapping a VapourSynth core.
    """

    def __init__(self, core: Core | None = None) -> None:
        object.__setattr__(self, "_own_core", core is not None)
        object.__setattr__(self, "_core_ref", core and weakref.ref(core))

    def __getattr__(self, name: str) -> Plugin:
        return getattr(self.core, name)

    def __setattr__(self, name: str, value: Any) -> None:
        return setattr(self.core, name, value)

    @property
    def env(self) -> EnvironmentProxy:
        """
        The EnvironmentProxy singleton representing the current environment execution context.

        Raises:
            CustomRuntimeError: If a policy has not been registered.
        """
        if not has_policy():
            raise CustomRuntimeError("No policy has been registered!")

        return _env_proxy

    @property
    def core_id(self) -> int:
        """
        The unique integer identifier of the active VapourSynth Core.

        Raises:
            CustomRuntimeError: If the Core has not been instantiated yet.
        """
        if not self.active:
            raise CustomRuntimeError("Core hasn't been fetched yet!")

        return id(self.core)

    @property
    def active(self) -> bool:
        """
        Check if the VapourSynth Core has already been instantiated.

        A core is active if we have a direct/owned Core instance,
        or if a registered policy environment exists and already has an initialized Core.
        """
        return (has_policy() and has_environment() and self.env.has_core) or (self._core is not None)

    @property
    def core(self) -> Core:
        """
        The underlying VapourSynth Core instance.
        """
        return self._core_with_cb

    @property
    def proxied(self) -> CoreProxy:
        """
        A `CoreProxy` backed by a weak reference to the current ``Core``.

        Plugins and functions are lazily resolved, so it's safe to hold references at class or module level
        (e.g. ``BlankClip = core.proxied.std.BlankClip``).

        If the underlying ``Core`` is freed and a new one is created, the proxy transparently falls back
        to the new core on next access.

        Accessing this property *may* trigger core creation if no core exists yet.
        """
        if self not in _objproxies:
            _objproxies[self] = {}

        if "proxied" not in _objproxies[self]:
            _objproxies[self]["proxied"] = CoreProxy(self._core, self, True)

        return _objproxies[self]["proxied"]

    @property
    def lazy(self) -> CoreProxy:
        """
        A `CoreProxy` with no initial ``Core`` reference.

        Like ``proxied``, plugins and functions are lazily resolved,
        but attribute access alone will *never* trigger core creation.
        A ``Core`` is only instantiated when a resolved function is actually invoked.

        Use this when you need to set class-level defaults.
        """
        if self not in _objproxies:
            _objproxies[self] = {}

        if "lazy" not in _objproxies[self]:
            _objproxies[self]["lazy"] = CoreProxy(None, self, True)

        return _objproxies[self]["lazy"]

    @deprecated("This function is deprecated. Use `vs.register_on_destroy()` instead.", category=DeprecationWarning)
    def register_on_destroy(self, callback: Callable[[], None]) -> None:
        register_on_destroy(callback)

    @deprecated("This function is deprecated. Use `vs.unregister_on_destroy()` instead.", category=DeprecationWarning)
    def unregister_on_destroy(self, callback: Callable[[], None]) -> None:
        unregister_on_destroy(callback)

    def set_affinity(
        self,
        threads: int | float | range | tuple[int, int] | list[int] | None = None,  # noqa: PYI041
        max_cache: int | None = None,
        reserve: Iterable[int] | None = None,
    ) -> None:
        """
        Configure CPU core affinity and cache settings for VapourSynth.

        This function selects which CPU cores the current process is allowed to run on,
        and configures the number of worker threads used by VapourSynth. It also allows
        tuning of the frame buffer cache.

        Args:
            threads: Defines how many and which CPU cores to use.

                Accepted formats:

                   - ``None``: Use all available CPU cores.
                   - ``int``: Use cores ``0`` through ``threads - 1``.
                   - ``float``: A fraction of available cores (e.g., ``0.5`` = half the cores).
                   - ``range``: Use the specified range of cores.
                   - ``tuple[int, int]``: Equivalent to ``range(start, stop)``.
                   - ``list[int]``: Explicit list of core indices.

            max_cache: Maximum VapourSynth frame buffer cache size, in megabytes.
                If ``None``, the default setting is preserved.

        Raises:
            CustomValueError: If ``threads`` is lower than or equal to 0.
        """
        import math
        import multiprocessing

        import psutil

        if threads is None:
            threads = multiprocessing.cpu_count()

        if isinstance(threads, float):
            if threads <= 0:
                raise CustomValueError(
                    "When passing a float, `threads` should be greater than 0.", self.set_affinity, threads
                )

            threads = math.ceil(multiprocessing.cpu_count() * threads)

        if isinstance(threads, int):
            threads = range(0, threads)
        elif isinstance(threads, tuple):
            threads = range(*threads)

        threads = list(set(threads) - set(reserve or []))

        self.core.num_threads = len(threads)

        psutil.Process().cpu_affinity(threads)

        if max_cache is not None:
            self.core.max_cache_size = max_cache

    @property
    def _core(self) -> Core | None:
        core_ref: weakref.ReferenceType[Core] | None = object.__getattribute__(self, "_core_ref")
        own_core: bool = object.__getattribute__(self, "_own_core")

        if core := (core_ref and core_ref()):
            return core

        if own_core:
            raise CustomRuntimeError("The core the proxy made reference to was freed!", "VSCoreProxy")

        return None

    @property
    def _core_with_cb(self) -> Core:
        """
        Retrieve the VapourSynth core and ensure all "on_creation" callbacks have been run.

        This makes sure callbacks are executed when a new core is created.
        """
        vs_core = self._core

        if not vs_core:
            import vapoursynth

            vs_core = vapoursynth.core.core

        # Map each core to the callbacks already run for it.
        # Weak references allow automatic cleanup when objects are destroyed.
        if vs_core not in _core_on_creation_callbacks_cores:
            _core_on_creation_callbacks_cores[vs_core] = weakref.WeakSet()

        run_cbs = _core_on_creation_callbacks_cores[vs_core]
        core_id = id(vs_core)

        # Run callbacks that have not yet been called for this core.
        for callback in list(_core_on_creation_callbacks):
            if callback in run_cbs:
                continue

            callback(core_id)
            # Remember that this callback was run for this core.
            run_cbs.add(callback)

        return vs_core


_core_on_creation_callbacks = weakref.WeakSet[Callable[[int], None]]()
_core_on_creation_callbacks_cores = weakref.WeakKeyDictionary[Core, weakref.WeakSet[Callable[[int], None]]]()
_env_proxy = EnvironmentProxy()
_objproxies = weakref.WeakKeyDictionary[VSCoreProxy, dict[Literal["proxied", "lazy"], CoreProxy]]()

core = VSCoreProxy()
"""The singleton Core object."""
