from __future__ import annotations

import builtins
import gc
import weakref

from ctypes import Structure
from inspect import Parameter, Signature
from logging import NOTSET as LogLevelUnset
from logging import Handler, LogRecord
from math import ceil
from multiprocessing import cpu_count
from types import UnionType
from typing import TYPE_CHECKING, Any, Callable, Iterable, NoReturn
from weakref import ReferenceType

import vapoursynth as vs

from vapoursynth import (
    AUDIO, BACK_CENTER, BACK_LEFT, BACK_RIGHT, CHROMA_BOTTOM, CHROMA_BOTTOM_LEFT, CHROMA_CENTER, CHROMA_LEFT,
    CHROMA_TOP, CHROMA_TOP_LEFT, DISABLE_AUTO_LOADING, DISABLE_LIBRARY_UNLOADING, ENABLE_GRAPH_INSPECTION, FIELD_BOTTOM,
    FIELD_PROGRESSIVE, FIELD_TOP, FLOAT, FRAME_STATE, FRONT_CENTER, FRONT_LEFT, FRONT_LEFT_OF_CENTER, FRONT_RIGHT,
    FRONT_RIGHT_OF_CENTER, GRAY, INTEGER, LOW_FREQUENCY, LOW_FREQUENCY2, MATRIX_BT470_BG, MATRIX_BT709,
    MATRIX_BT2020_CL, MATRIX_BT2020_NCL, MATRIX_CHROMATICITY_DERIVED_CL, MATRIX_CHROMATICITY_DERIVED_NCL, MATRIX_FCC,
    MATRIX_ICTCP, MATRIX_RGB, MATRIX_ST170_M, MATRIX_UNSPECIFIED, MATRIX_YCGCO, MESSAGE_TYPE_CRITICAL,
    MESSAGE_TYPE_DEBUG, MESSAGE_TYPE_FATAL, MESSAGE_TYPE_INFORMATION, MESSAGE_TYPE_WARNING, NONE, PARALLEL,
    PARALLEL_REQUESTS, PRIMARIES_BT470_BG, PRIMARIES_BT470_M, PRIMARIES_BT709, PRIMARIES_BT2020, PRIMARIES_EBU3213_E,
    PRIMARIES_FILM, PRIMARIES_ST170_M, PRIMARIES_ST240_M, PRIMARIES_ST428, PRIMARIES_ST431_2, PRIMARIES_ST432_1,
    PRIMARIES_UNSPECIFIED, RANGE_FULL, RANGE_LIMITED, RGB, SIDE_LEFT, SIDE_RIGHT, STEREO_LEFT, STEREO_RIGHT,
    SURROUND_DIRECT_LEFT, SURROUND_DIRECT_RIGHT, TOP_BACK_CENTER, TOP_BACK_LEFT, TOP_BACK_RIGHT, TOP_CENTER,
    TOP_FRONT_CENTER, TOP_FRONT_LEFT, TOP_FRONT_RIGHT, TRANSFER_ARIB_B67, TRANSFER_BT470_BG, TRANSFER_BT470_M,
    TRANSFER_BT601, TRANSFER_BT709, TRANSFER_BT2020_10, TRANSFER_BT2020_12, TRANSFER_IEC_61966_2_1,
    TRANSFER_IEC_61966_2_4, TRANSFER_LINEAR, TRANSFER_LOG_100, TRANSFER_LOG_316, TRANSFER_ST240_M, TRANSFER_ST2084,
    TRANSFER_UNSPECIFIED, UNDEFINED, UNORDERED, VIDEO, WIDE_LEFT, WIDE_RIGHT, YUV, AudioChannels, AudioFrame, AudioNode,
    ChromaLocation, ColorFamily, ColorPrimaries, ColorRange, Core, CoreCreationFlags, Environment, EnvironmentData,
    EnvironmentPolicy, EnvironmentPolicyAPI, Error, FieldBased, FilterMode, FrameProps, FramePtr, Func, FuncData,
    Function, LogHandle, MatrixCoefficients, MediaType, MessageType, Plugin, RawFrame, RawNode, SampleType,
    TransferCharacteristics, VideoFormat, VideoFrame, VideoNode, VideoOutputTuple, __api_version__, __version__,
    _CoreProxy, clear_output, clear_outputs, get_current_environment, get_output, get_outputs, has_policy,
    register_on_destroy, register_policy, unregister_on_destroy
)

from ..exceptions import CustomRuntimeError
from .vs_enums import (
    GRAY8, GRAY9, GRAY10, GRAY11, GRAY12, GRAY13, GRAY14, GRAY15, GRAY16, GRAY17, GRAY18, GRAY19, GRAY20, GRAY21,
    GRAY22, GRAY23, GRAY24, GRAY25, GRAY26, GRAY27, GRAY28, GRAY29, GRAY30, GRAY31, GRAY32, GRAYH, GRAYS, RGB24, RGB27,
    RGB30, RGB33, RGB36, RGB39, RGB42, RGB45, RGB48, RGB51, RGB54, RGB57, RGB60, RGB63, RGB66, RGB69, RGB72, RGB75,
    RGB78, RGB81, RGB84, RGB87, RGB90, RGB93, RGB96, RGBH, RGBS, YUV410P8, YUV410P9, YUV410P10, YUV410P11, YUV410P12,
    YUV410P13, YUV410P14, YUV410P15, YUV410P16, YUV410P17, YUV410P18, YUV410P19, YUV410P20, YUV410P21, YUV410P22,
    YUV410P23, YUV410P24, YUV410P25, YUV410P26, YUV410P27, YUV410P28, YUV410P29, YUV410P30, YUV410P31, YUV410P32,
    YUV410PH, YUV410PS, YUV411P8, YUV411P9, YUV411P10, YUV411P11, YUV411P12, YUV411P13, YUV411P14, YUV411P15, YUV411P16,
    YUV411P17, YUV411P18, YUV411P19, YUV411P20, YUV411P21, YUV411P22, YUV411P23, YUV411P24, YUV411P25, YUV411P26,
    YUV411P27, YUV411P28, YUV411P29, YUV411P30, YUV411P31, YUV411P32, YUV411PH, YUV411PS, YUV420P8, YUV420P9, YUV420P10,
    YUV420P11, YUV420P12, YUV420P13, YUV420P14, YUV420P15, YUV420P16, YUV420P17, YUV420P18, YUV420P19, YUV420P20,
    YUV420P21, YUV420P22, YUV420P23, YUV420P24, YUV420P25, YUV420P26, YUV420P27, YUV420P28, YUV420P29, YUV420P30,
    YUV420P31, YUV420P32, YUV420PH, YUV420PS, YUV422P8, YUV422P9, YUV422P10, YUV422P11, YUV422P12, YUV422P13, YUV422P14,
    YUV422P15, YUV422P16, YUV422P17, YUV422P18, YUV422P19, YUV422P20, YUV422P21, YUV422P22, YUV422P23, YUV422P24,
    YUV422P25, YUV422P26, YUV422P27, YUV422P28, YUV422P29, YUV422P30, YUV422P31, YUV422P32, YUV422PH, YUV422PS,
    YUV440P8, YUV440P9, YUV440P10, YUV440P11, YUV440P12, YUV440P13, YUV440P14, YUV440P15, YUV440P16, YUV440P17,
    YUV440P18, YUV440P19, YUV440P20, YUV440P21, YUV440P22, YUV440P23, YUV440P24, YUV440P25, YUV440P26, YUV440P27,
    YUV440P28, YUV440P29, YUV440P30, YUV440P31, YUV440P32, YUV440PH, YUV440PS, YUV444P8, YUV444P9, YUV444P10, YUV444P11,
    YUV444P12, YUV444P13, YUV444P14, YUV444P15, YUV444P16, YUV444P17, YUV444P18, YUV444P19, YUV444P20, YUV444P21,
    YUV444P22, YUV444P23, YUV444P24, YUV444P25, YUV444P26, YUV444P27, YUV444P28, YUV444P29, YUV444P30, YUV444P31,
    YUV444P32, YUV444PH, YUV444PS, PresetVideoFormat, VSPresetVideoFormat
)

__all__ = [
    'AUDIO', 'AudioChannels', 'AudioFrame', 'AudioNode', 'BACK_CENTER', 'BACK_LEFT', 'BACK_RIGHT', 'CHROMA_BOTTOM',
    'CHROMA_BOTTOM_LEFT', 'CHROMA_CENTER', 'CHROMA_LEFT', 'CHROMA_TOP', 'CHROMA_TOP_LEFT', 'CallbackData',
    'ChromaLocation', 'ColorFamily', 'ColorPrimaries', 'ColorRange', 'Core', 'CoreCreationFlags', 'Environment',
    'EnvironmentData', 'EnvironmentPolicy', 'EnvironmentPolicyAPI', 'Error', 'FIELD_BOTTOM', 'FIELD_PROGRESSIVE',
    'FIELD_TOP', 'FLOAT', 'FRONT_CENTER', 'FRONT_LEFT', 'FRONT_LEFT_OF_CENTER', 'FRONT_RIGHT', 'FRONT_RIGHT_OF_CENTER',
    'FieldBased', 'FilterMode', 'FrameProps', 'FramePtr', 'Func', 'FuncData', 'Function', 'GRAY', 'GRAY10', 'GRAY11',
    'GRAY12', 'GRAY13', 'GRAY14', 'GRAY15', 'GRAY16', 'GRAY17', 'GRAY18', 'GRAY19', 'GRAY20', 'GRAY21', 'GRAY22',
    'GRAY23', 'GRAY24', 'GRAY25', 'GRAY26', 'GRAY27', 'GRAY28', 'GRAY29', 'GRAY30', 'GRAY31', 'GRAY32', 'GRAY8',
    'GRAY9', 'GRAYH', 'GRAYS', 'INTEGER', 'LOW_FREQUENCY', 'LOW_FREQUENCY2', 'LogHandle', 'MATRIX_BT2020_CL',
    'MATRIX_BT2020_NCL', 'MATRIX_BT470_BG', 'MATRIX_BT709', 'MATRIX_CHROMATICITY_DERIVED_CL',
    'MATRIX_CHROMATICITY_DERIVED_NCL', 'MATRIX_FCC', 'MATRIX_ICTCP', 'MATRIX_RGB', 'MATRIX_ST170_M',
    'MATRIX_UNSPECIFIED', 'MATRIX_YCGCO', 'MESSAGE_TYPE_CRITICAL', 'MESSAGE_TYPE_DEBUG', 'MESSAGE_TYPE_FATAL',
    'MESSAGE_TYPE_INFORMATION', 'MESSAGE_TYPE_WARNING', 'MatrixCoefficients', 'MediaType', 'MessageType', 'NONE',
    'PRIMARIES_BT2020', 'PRIMARIES_BT470_BG', 'PRIMARIES_BT470_M', 'PRIMARIES_BT709', 'PRIMARIES_EBU3213_E',
    'PRIMARIES_FILM', 'PRIMARIES_ST170_M', 'PRIMARIES_ST240_M', 'PRIMARIES_ST428', 'PRIMARIES_ST431_2',
    'PRIMARIES_ST432_1', 'PRIMARIES_UNSPECIFIED', 'Plugin', 'PresetVideoFormat',
    'PythonVSScriptLoggingBridge', 'RANGE_FULL',
    'RANGE_LIMITED', 'RGB', 'RGB24', 'RGB27', 'RGB30', 'RGB33', 'RGB36', 'RGB39', 'RGB42', 'RGB45', 'RGB48', 'RGB51',
    'RGB54', 'RGB57', 'RGB60', 'RGB63', 'RGB66', 'RGB69', 'RGB72', 'RGB75', 'RGB78', 'RGB81', 'RGB84', 'RGB87', 'RGB90',
    'RGB93', 'RGB96', 'RGBH', 'RGBS', 'RawFrame', 'RawNode', 'SIDE_LEFT', 'SIDE_RIGHT', 'STEREO_LEFT', 'STEREO_RIGHT',
    'SURROUND_DIRECT_LEFT', 'SURROUND_DIRECT_RIGHT', 'SampleType', 'StandaloneEnvironmentPolicy', 'TOP_BACK_CENTER',
    'TOP_BACK_LEFT', 'TOP_BACK_RIGHT', 'TOP_CENTER', 'TOP_FRONT_CENTER', 'TOP_FRONT_LEFT', 'TOP_FRONT_RIGHT',
    'TRANSFER_ARIB_B67', 'TRANSFER_BT2020_10', 'TRANSFER_BT2020_12', 'TRANSFER_BT470_BG', 'TRANSFER_BT470_M',
    'TRANSFER_BT601', 'TRANSFER_BT709', 'TRANSFER_IEC_61966_2_1', 'TRANSFER_IEC_61966_2_4', 'TRANSFER_LINEAR',
    'TRANSFER_LOG_100', 'TRANSFER_LOG_316', 'TRANSFER_ST2084', 'TRANSFER_ST240_M', 'TRANSFER_UNSPECIFIED',
    'TransferCharacteristics', 'UNDEFINED', 'VIDEO', 'VSScriptEnvironmentPolicy', 'VideoFormat', 'VideoFrame',
    'VideoNode', 'VideoOutputTuple', 'WIDE_LEFT', 'WIDE_RIGHT', 'YUV', 'YUV410P10', 'YUV410P11', 'YUV410P12',
    'YUV410P13', 'YUV410P14', 'YUV410P15', 'YUV410P16', 'YUV410P17', 'YUV410P18', 'YUV410P19', 'YUV410P20', 'YUV410P21',
    'YUV410P22', 'YUV410P23', 'YUV410P24', 'YUV410P25', 'YUV410P26', 'YUV410P27', 'YUV410P28', 'YUV410P29', 'YUV410P30',
    'YUV410P31', 'YUV410P32', 'YUV410P8', 'YUV410P9', 'YUV410PH', 'YUV410PS', 'YUV411P10', 'YUV411P11', 'YUV411P12',
    'YUV411P13', 'YUV411P14', 'YUV411P15', 'YUV411P16', 'YUV411P17', 'YUV411P18', 'YUV411P19', 'YUV411P20', 'YUV411P21',
    'YUV411P22', 'YUV411P23', 'YUV411P24', 'YUV411P25', 'YUV411P26', 'YUV411P27', 'YUV411P28', 'YUV411P29', 'YUV411P30',
    'YUV411P31', 'YUV411P32', 'YUV411P8', 'YUV411P9', 'YUV411PH', 'YUV411PS', 'YUV420P10', 'YUV420P11', 'YUV420P12',
    'YUV420P13', 'YUV420P14', 'YUV420P15', 'YUV420P16', 'YUV420P17', 'YUV420P18', 'YUV420P19', 'YUV420P20', 'YUV420P21',
    'YUV420P22', 'YUV420P23', 'YUV420P24', 'YUV420P25', 'YUV420P26', 'YUV420P27', 'YUV420P28', 'YUV420P29', 'YUV420P30',
    'YUV420P31', 'YUV420P32', 'YUV420P8', 'YUV420P9', 'YUV420PH', 'YUV420PS', 'YUV422P10', 'YUV422P11', 'YUV422P12',
    'YUV422P13', 'YUV422P14', 'YUV422P15', 'YUV422P16', 'YUV422P17', 'YUV422P18', 'YUV422P19', 'YUV422P20', 'YUV422P21',
    'YUV422P22', 'YUV422P23', 'YUV422P24', 'YUV422P25', 'YUV422P26', 'YUV422P27', 'YUV422P28', 'YUV422P29', 'YUV422P30',
    'YUV422P31', 'YUV422P32', 'YUV422P8', 'YUV422P9', 'YUV422PH', 'YUV422PS', 'YUV440P10', 'YUV440P11', 'YUV440P12',
    'YUV440P13', 'YUV440P14', 'YUV440P15', 'YUV440P16', 'YUV440P17', 'YUV440P18', 'YUV440P19', 'YUV440P20', 'YUV440P21',
    'YUV440P22', 'YUV440P23', 'YUV440P24', 'YUV440P25', 'YUV440P26', 'YUV440P27', 'YUV440P28', 'YUV440P29', 'YUV440P30',
    'YUV440P31', 'YUV440P32', 'YUV440P8', 'YUV440P9', 'YUV440PH', 'YUV440PS', 'YUV444P10', 'YUV444P11', 'YUV444P12',
    'YUV444P13', 'YUV444P14', 'YUV444P15', 'YUV444P16', 'YUV444P17', 'YUV444P18', 'YUV444P19', 'YUV444P20', 'YUV444P21',
    'YUV444P22', 'YUV444P23', 'YUV444P24', 'YUV444P25', 'YUV444P26', 'YUV444P27', 'YUV444P28', 'YUV444P29', 'YUV444P30',
    'YUV444P31', 'YUV444P32', 'YUV444P8', 'YUV444P9', 'YUV444PH', 'YUV444PS', '_CoreProxy', '__all__',
    '__api_version__', '__version__', 'CoreCreationFlags', 'ENABLE_GRAPH_INSPECTION', 'DISABLE_AUTO_LOADING',
    'DISABLE_LIBRARY_UNLOADING', 'clear_output', 'clear_outputs', 'construct_parameter', 'construct_signature',
    'construct_type', 'core', 'FilterMode', 'PARALLEL', 'PARALLEL_REQUESTS', 'UNORDERED', 'FRAME_STATE',
    'get_current_environment', 'get_output', 'get_outputs', 'has_policy',
    'pyx_capi', 'register_on_creation', 'register_on_destroy', 'register_policy', 'try_enable_introspection',
    'unregister_on_creation', 'unregister_on_destroy', 'vs_file', 'clear_cache'
]


if not TYPE_CHECKING:
    import inspect
    import re
    import sys
    import warnings

    from pathlib import Path
    from types import ModuleType

    import __main__

    if not hasattr(__main__, '__file__') and '__vapoursynth__' not in sys.modules:
        first_stack = inspect.stack()[-1]

        sys.modules['__vapoursynth__'] = ModuleType('__vapoursynth__')

        cope = (Path.cwd() / first_stack.filename).resolve()

        first_stack = None

        sys.modules['__vapoursynth__'].__file__ = __main__.__file__ = str(cope)

        sys.path.append(str(cope.parent))

    warnings._add_filter('ignore', re.compile('.*smallest subnormal.*numpy.*'), Warning, None, 0, append=False)
    warnings._add_filter('ignore', re.compile('.*divide by zero.*'), Warning, None, 0, append=False)


def register_on_creation(callback: Callable[..., None], strict: bool = False) -> None:
    """Register a callback on every core creation."""

    core_on_creation_callbacks.update({id(callback): weakref.ref(callback)})

    if not strict and core.active:
        try:
            callback(core.core_id)
        except TypeError:
            callback()


def unregister_on_creation(callback: Callable[..., None]) -> None:
    """Unregister this callback from every core creation."""

    core_on_creation_callbacks.pop(id(callback), None)


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


if TYPE_CHECKING:
    class FunctionProxyBase(Function):
        ...

    class PluginProxyBase(Plugin):
        ...

    class CoreProxyBase(Core):
        def __init__(self) -> None:
            ...

    class EnvironmentProxyBase(Environment):
        def __init__(self) -> None:
            ...
else:
    FunctionProxyBase = PluginProxyBase = CoreProxyBase = EnvironmentProxyBase = object


class FunctionProxy(FunctionProxyBase):
    def __init__(self, plugin: PluginProxy, func_name: str) -> None:
        self.__dict__['func_ref'] = (plugin, func_name)

    def __getattr__(self, name: str) -> Function:
        if name == '__isabstractmethod__':
            return False  # type: ignore[return-value]

        function = proxy_utils.get_vs_function(self)

        return getattr(function, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return proxy_utils.get_vs_function(self)(*args, **kwargs)


class PluginProxy(PluginProxyBase):
    def __init__(self, core: CoreProxy, namespace: str) -> None:
        self.__dict__['plugin_ref'] = (core, namespace)

    def __getattr__(self, name: str) -> Function:
        core, namespace = proxy_utils.get_core(self)

        if core.lazy and name not in vs.Plugin.__dict__:
            return FunctionProxy(self, name)

        vs_core = proxy_utils.get_vs_core(core)

        plugin = getattr(vs_core, namespace)

        if name in dir(plugin):
            return FunctionProxy(self, name)

        return getattr(plugin, name)


class CoreProxy(CoreProxyBase):
    def __init__(self, core: Core | None, vs_proxy: VSCoreProxy, lazy: bool) -> None:
        self.lazy = lazy
        self.__dict__['vs_core_ref'] = (core and weakref.ref(core), vs_proxy)

    def __getattr__(self, name: str) -> Plugin:
        if self.lazy and name not in vs.Core.__dict__:
            return PluginProxy(self, name)

        core = proxy_utils.get_vs_core(self)

        if name in dir(core):
            return PluginProxy(self, name)

        return getattr(core, name)


class proxy_utils:
    @staticmethod
    def get_vs_core(core: CoreProxy) -> Core:
        vs_core_ref, vs_proxy = core.__dict__['vs_core_ref']

        vs_core = (vs_core_ref and vs_core_ref())

        if vs_core_ref and vs_core is None:
            if object.__getattribute__(vs_proxy, '_own_core'):
                raise CustomRuntimeError('The VapourSynth core has been freed!', CoreProxy)

            vs_core = _get_core(vs_proxy)
            core.__dict__['vs_core_ref'] = (vs_core and weakref.ref(vs_core), vs_proxy)

        return vs_core or _get_core_with_cb()

    @staticmethod
    def get_vs_function(func: FunctionProxy) -> Function:
        plugin, func_name = proxy_utils.get_plugin(func)
        core, namespace = proxy_utils.get_core(plugin)
        vs_core = proxy_utils.get_vs_core(core)

        return getattr(getattr(vs_core, namespace), func_name)

    @staticmethod
    def get_plugin(func: FunctionProxy) -> tuple[PluginProxy, str]:
        return func.__dict__['func_ref']

    @staticmethod
    def get_core(plugin: PluginProxy) -> tuple[CoreProxy, str]:
        return plugin.__dict__['plugin_ref']


def vstools_isinstance(
    __obj: object, __class_or_tuple: type | UnionType | tuple[type | UnionType | tuple[Any, ...], ...]
) -> bool:
    if __class_or_tuple in (_CoreProxy, Core) and builtins_isinstance(__obj, CoreProxy):
        return True

    if __class_or_tuple is VSPresetVideoFormat and builtins_isinstance(__obj, PresetVideoFormat):
        return True

    return builtins_isinstance(__obj, __class_or_tuple)


if builtins.isinstance is not vstools_isinstance:
    builtins_isinstance = builtins.isinstance
    builtins.isinstance = vstools_isinstance


def _get_core(self: VSCoreProxy) -> Core | None:
    core_ref: ReferenceType[Core] | None = object.__getattribute__(self, '_core')
    own_core: bool = object.__getattribute__(self, '_own_core')

    if core := (core_ref and core_ref()):
        return core

    if own_core:
        raise CustomRuntimeError(
            'The core the proxy made reference to was freed!', 'VSCoreProxy'
        )

    return None


if TYPE_CHECKING:
    core_on_creation_callbacks = dict[int, weakref.ReferenceType[Callable[..., None]]]()
else:
    core_on_creation_callbacks = {}

core_on_creation_callbacks_cores = set[int]()


def _get_core_with_cb(self: VSCoreProxy | None = None) -> Core:
    _vs_core = _get_core(self) if self else None

    if not _vs_core:
        _vs_core = vs.core.core

    if (core_id := id(_vs_core)) not in core_on_creation_callbacks_cores:
        for cb_id in list(core_on_creation_callbacks.keys()):
            callback_ref = core_on_creation_callbacks.get(cb_id, None)

            if callback_ref and (callback := callback_ref()):
                try:
                    callback(core_id)
                except TypeError:
                    callback()
            else:
                # remove dead references
                core_on_creation_callbacks.pop(cb_id, None)

        core_on_creation_callbacks_cores.add(id(_vs_core))

    return _vs_core


def _find_ref(start_data: Any, to_return: type | tuple[type, ...], it: int = 3) -> Any:
    if not it:
        return None

    for objects in [gc.get_referents(start_data), gc.get_referrers(start_data)]:
        for obj in objects:
            if isinstance(obj, to_return):
                return obj

            if (isinstance(obj, dict) and '__name__' in obj):
                continue

            if isinstance(obj, (Core, _CoreProxy, _FastManager)):
                continue

            for obj_obj in gc.get_referents(obj):
                if isinstance(obj_obj, to_return):
                    return obj_obj

                value = _find_ref(obj, to_return, it - 1)

                if value:
                    return value

    return None


class EnvironmentProxy(EnvironmentProxyBase):
    def __getattr__(self, name: str) -> Plugin:
        return getattr(get_current_environment(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        return setattr(get_current_environment(), name, value)

    @property
    def data(self) -> None:
        data = self.env()
        assert data
        return data  # type: ignore

    @property
    def policy(self) -> EnvironmentPolicy:
        policy = _find_ref(self.data, (EnvironmentPolicy, VSScriptEnvironmentPolicy, StandaloneEnvironmentPolicy))
        assert policy is not None
        return policy

    @property
    def api(self) -> EnvironmentPolicyAPI:
        api = _find_ref(self.policy, EnvironmentPolicyAPI)
        assert api is not None
        return api

    @property
    def has_core(self) -> bool:
        import gc

        found_core = _find_ref(self.data, Core)

        if not found_core:
            return False

        core_refs = gc.get_referrers(found_core)

        return len(core_refs) > 1


_curr_env_proxy = EnvironmentProxy()


class VSCoreProxy(CoreProxyBase):
    """Class for wrapping a VapourSynth core."""

    def __init__(self, core: Core | None = None) -> None:
        object.__setattr__(self, '_own_core', core is not None)
        object.__setattr__(self, '_core', core and weakref.ref(core))

    def __getattr__(self, name: str) -> Plugin:
        return getattr(_get_core_with_cb(self), name)

    def __setattr__(self, name: str, value: Any) -> None:
        return setattr(_get_core_with_cb(self), name, value)

    @property
    def env(self) -> EnvironmentProxy:
        if not has_policy():
            raise CustomRuntimeError('No policy has been registered!')

        return _curr_env_proxy

    @property
    def core_id(self) -> int:
        if not self.active:
            raise CustomRuntimeError("Core hasn't been fetched yet!")

        return id(self.core)

    @property
    def active(self) -> bool:
        return (has_policy() and self.env.has_core) or (_get_core(self) is not None)

    @property
    def core(self) -> Core:
        """The underlying VapourSynth Core instance."""

        return _get_core_with_cb(self)

    @property
    def proxied(self) -> CoreProxy:
        """
        Proxied Core where plugins and functions are lazily retrieved,
        so it's safe to hold a reference of anything from this.
        """

        if self not in _objproxies:
            _objproxies[self] = {}

        if 'proxied' not in _objproxies[self]:
            _objproxies[self]['proxied'] = CoreProxy(_get_core(self), self, True)

        return _objproxies[self]['proxied']

    @property
    def lazy(self) -> CoreProxy:
        """
        Lazy Core where plugins and functions are lazily retrieved and checked,
        so it's safe to hold a reference and set default of anything from this,
        without having to worry of creating a core.
        """

        if self not in _objproxies:
            _objproxies[self] = {}

        if 'lazy' not in _objproxies[self]:
            _objproxies[self]['lazy'] = CoreProxy(None, self, True)

        return _objproxies[self]['lazy']

    def register_on_destroy(self, callback: Callable[..., None], on_forced: bool = True) -> None:
        """Register a callback on this core destroy."""

        _check_environment()
        register_on_destroy(callback)

    def unregister_on_destroy(self, callback: Callable[..., None]) -> None:
        """Unregister a callback from this core destroy."""

        _check_environment()
        unregister_on_destroy(callback)

    def set_affinity(
        self, threads: int | float | range | tuple[int, int] | list[int] | None = None,
        max_cache: int | None = None, reserve: int | Iterable[int] = 2
    ) -> None:
        from psutil import Process

        """
        Set core affinity.

        :param threads:     How many and which threads to use for VapourSynth.
        :param max_cache:   Maximum cache used for frame data in VapourSynth.
        :param reserve:     Reserve n amount of or the specified threads.

        :raises DependencyNotFoundError:    Psutil was not found.
        """

        if threads is None:
            threads = cpu_count()

        if isinstance(threads, float):
            if 0.0 <= threads or threads >= 1.0:
                threads = 1.0

            threads = ceil(cpu_count() * threads)

        if isinstance(threads, int):
            threads = range(0, threads)
        elif isinstance(threads, tuple):
            threads = range(*threads)

        threads = list(set(threads))

        if isinstance(reserve, int):
            if reserve > len(threads):
                threads = threads[:-reserve]
        else:
            threads = [t for t in threads if t not in reserve]

        self.core.num_threads = len(threads)

        Process().cpu_affinity(threads)

        if max_cache is not None:
            self.core.max_cache_size = max_cache


def _core_on_destroy_try() -> None:
    ...


def _check_environment() -> None:
    try:
        register_on_destroy(_core_on_destroy_try)
        unregister_on_destroy(_core_on_destroy_try)
    except Exception as e:
        if isinstance(e, ValueError) or not get_current_environment().active:
            raise ValueError("The environment has already been destroyed.")


_objproxies = {}  # type: ignore

core = VSCoreProxy()


if TYPE_CHECKING:
    class PyCapsule(Structure):
        ...

    pyx_capi: dict[str, PyCapsule] = ...  # type: ignore

    class StandaloneEnvironmentPolicy(EnvironmentPolicy):
        def __init__(self) -> NoReturn:
            ...

        def _on_log_message(self, level: MessageType, msg: str) -> None:
            ...

        def on_policy_registered(self, api: 'EnvironmentPolicyAPI') -> None:
            ...

        def on_policy_cleared(self) -> None:
            ...

        def get_current_environment(self) -> EnvironmentData:
            ...

        def set_environment(self, environment: EnvironmentData | None) -> EnvironmentData:
            ...

        def is_alive(self, environment: EnvironmentData) -> bool:
            ...

    class VSScriptEnvironmentPolicy(EnvironmentPolicy):
        def __init__(self) -> NoReturn:
            ...

        def on_policy_registered(self, policy_api: 'EnvironmentPolicyAPI') -> None:
            ...

        def on_policy_cleared(self) -> None:
            ...

        def get_current_environment(self) -> EnvironmentData | None:
            ...

        def set_environment(self, environment: EnvironmentData | None) -> EnvironmentData | None:
            ...

        def is_alive(self, environment: EnvironmentData) -> bool:
            ...

    def construct_type(signature: str) -> type:
        ...

    def construct_parameter(signature: str) -> Parameter:
        ...

    def construct_signature(signature: str, return_signature: str, injected: str | None = None) -> Signature:
        ...

    def try_enable_introspection(version: int | None = None) -> bool:  # noqa: F811
        ...

    class CallbackData:
        def __init__(
            self, node: RawNode, env: EnvironmentData,
            callback: Callable[[RawFrame | None, Exception | None], None] | None = None
        ) -> None:
            ...

        def receive(self, n: int, result: RawFrame | Exception) -> None:
            ...

    class PythonVSScriptLoggingBridge(Handler):
        def __init__(self, parent: Handler, level: int = LogLevelUnset) -> None:
            ...

        def emit(self, record: LogRecord) -> None:
            ...

    class _FastManager:
        ...
else:
    from vapoursynth import (
        CallbackData, PythonVSScriptLoggingBridge, StandaloneEnvironmentPolicy, VSScriptEnvironmentPolicy
    )
    from vapoursynth import __file__ as vs_file
    from vapoursynth import __pyx_capi__ as pyx_capi
    from vapoursynth import _construct_parameter as construct_parameter
    from vapoursynth import _construct_type as construct_type
    from vapoursynth import _FastManager
    from vapoursynth import _try_enable_introspection as try_enable_introspection
    from vapoursynth import construct_signature
