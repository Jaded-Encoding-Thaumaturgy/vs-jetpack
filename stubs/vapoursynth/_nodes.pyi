from fractions import Fraction
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Callable,
    Generic,
    Iterator,
    Literal,
    Protocol,
    Self,
    TypedDict,
    TypeVar,
    overload,
)

if TYPE_CHECKING:
    from vstools import ConstantFormatVideoNode

from ._enums import ColorFamily, FilterMode, MessageType, PresetVideoFormat, SampleType
from ._formats import ChannelLayout, VideoFormat
from ._frames import AudioFrame, RawFrame, VideoFrame
from ._functions import Plugin
from ._logging import LogHandle
from ._typings import _DataType, _Future, _SingleAndSequence, _VapourSynthMapValue, _VSMapValueCallback

__all__ = [
    "RawNode",
    "VideoNode",
    "AudioNode",
    "Core",
    "_CoreProxy",
    "core",
]

# implementation: adg

class _Plugin_adg_Core_Bound(Plugin):
    """This class implements the module definitions for the "adg" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Mask(self, clip: VideoNode, luma_scaling: float | None = None) -> ConstantFormatVideoNode: ...

class _Plugin_adg_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "adg" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Mask(self, luma_scaling: float | None = None) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: akarin

_ReturnDict_akarin_Version = TypedDict(
    "_ReturnDict_akarin_Version",
    {
        "version": bytes,
        "expr_backend": bytes,
        "expr_features": list[bytes],
        "select_features": list[bytes],
        "text_features": list[bytes],
        "tmpl_features": list[bytes],
    },
)

class _Plugin_akarin_Core_Bound(Plugin):
    """This class implements the module definitions for the "akarin" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Cambi(
        self,
        clip: VideoNode,
        window_size: int | None = None,
        topk: float | None = None,
        tvi_threshold: float | None = None,
        scores: int | None = None,
        scaling: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def DLISR(
        self, clip: VideoNode, scale: int | None = None, device_id: int | None = None
    ) -> ConstantFormatVideoNode: ...
    def DLVFX(
        self,
        clip: VideoNode,
        op: int,
        scale: float | None = None,
        strength: float | None = None,
        output_depth: int | None = None,
        num_streams: int | None = None,
        model_dir: _DataType | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Expr(
        self,
        clips: _SingleAndSequence[VideoNode],
        expr: _SingleAndSequence[_DataType],
        format: int | None = None,
        opt: int | None = None,
        boundary: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def ExprTest(
        self,
        clips: _SingleAndSequence[float],
        expr: _DataType,
        props: _VSMapValueCallback[_VapourSynthMapValue] | None = None,
        ref: VideoNode | None = None,
        vars: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def PickFrames(self, clip: VideoNode, indices: _SingleAndSequence[int]) -> ConstantFormatVideoNode: ...
    def PropExpr(
        self, clips: _SingleAndSequence[VideoNode], dict: _VSMapValueCallback[_VapourSynthMapValue]
    ) -> ConstantFormatVideoNode: ...
    def Select(
        self,
        clip_src: _SingleAndSequence[VideoNode],
        prop_src: _SingleAndSequence[VideoNode],
        expr: _SingleAndSequence[_DataType],
    ) -> ConstantFormatVideoNode: ...
    def Text(
        self,
        clips: _SingleAndSequence[VideoNode],
        text: _DataType,
        alignment: int | None = None,
        scale: int | None = None,
        prop: _DataType | None = None,
        strict: int | None = None,
        vspipe: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Tmpl(
        self,
        clips: _SingleAndSequence[VideoNode],
        prop: _SingleAndSequence[_DataType],
        text: _SingleAndSequence[_DataType],
    ) -> ConstantFormatVideoNode: ...
    def Version(self) -> _ReturnDict_akarin_Version: ...

class _Plugin_akarin_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "akarin" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Cambi(
        self,
        window_size: int | None = None,
        topk: float | None = None,
        tvi_threshold: float | None = None,
        scores: int | None = None,
        scaling: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def DLISR(self, scale: int | None = None, device_id: int | None = None) -> ConstantFormatVideoNode: ...
    def DLVFX(
        self,
        op: int,
        scale: float | None = None,
        strength: float | None = None,
        output_depth: int | None = None,
        num_streams: int | None = None,
        model_dir: _DataType | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Expr(
        self,
        expr: _SingleAndSequence[_DataType],
        format: int | None = None,
        opt: int | None = None,
        boundary: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def PickFrames(self, indices: _SingleAndSequence[int]) -> ConstantFormatVideoNode: ...
    def PropExpr(self, dict: _VSMapValueCallback[_VapourSynthMapValue]) -> ConstantFormatVideoNode: ...
    def Select(
        self, prop_src: _SingleAndSequence[VideoNode], expr: _SingleAndSequence[_DataType]
    ) -> ConstantFormatVideoNode: ...
    def Text(
        self,
        text: _DataType,
        alignment: int | None = None,
        scale: int | None = None,
        prop: _DataType | None = None,
        strict: int | None = None,
        vspipe: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Tmpl(
        self, prop: _SingleAndSequence[_DataType], text: _SingleAndSequence[_DataType]
    ) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: bm3d

class _Plugin_bm3d_Core_Bound(Plugin):
    """This class implements the module definitions for the "bm3d" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Basic(
        self,
        input: VideoNode,
        ref: VideoNode | None = None,
        profile: _DataType | None = None,
        sigma: _SingleAndSequence[float] | None = None,
        block_size: int | None = None,
        block_step: int | None = None,
        group_size: int | None = None,
        bm_range: int | None = None,
        bm_step: int | None = None,
        th_mse: float | None = None,
        hard_thr: float | None = None,
        matrix: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Final(
        self,
        input: VideoNode,
        ref: VideoNode,
        profile: _DataType | None = None,
        sigma: _SingleAndSequence[float] | None = None,
        block_size: int | None = None,
        block_step: int | None = None,
        group_size: int | None = None,
        bm_range: int | None = None,
        bm_step: int | None = None,
        th_mse: float | None = None,
        matrix: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def OPP2RGB(self, input: VideoNode, sample: int | None = None) -> ConstantFormatVideoNode: ...
    def RGB2OPP(self, input: VideoNode, sample: int | None = None) -> ConstantFormatVideoNode: ...
    def VAggregate(
        self, input: VideoNode, radius: int | None = None, sample: int | None = None
    ) -> ConstantFormatVideoNode: ...
    def VBasic(
        self,
        input: VideoNode,
        ref: VideoNode | None = None,
        profile: _DataType | None = None,
        sigma: _SingleAndSequence[float] | None = None,
        radius: int | None = None,
        block_size: int | None = None,
        block_step: int | None = None,
        group_size: int | None = None,
        bm_range: int | None = None,
        bm_step: int | None = None,
        ps_num: int | None = None,
        ps_range: int | None = None,
        ps_step: int | None = None,
        th_mse: float | None = None,
        hard_thr: float | None = None,
        matrix: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def VFinal(
        self,
        input: VideoNode,
        ref: VideoNode,
        profile: _DataType | None = None,
        sigma: _SingleAndSequence[float] | None = None,
        radius: int | None = None,
        block_size: int | None = None,
        block_step: int | None = None,
        group_size: int | None = None,
        bm_range: int | None = None,
        bm_step: int | None = None,
        ps_num: int | None = None,
        ps_range: int | None = None,
        ps_step: int | None = None,
        th_mse: float | None = None,
        matrix: int | None = None,
    ) -> ConstantFormatVideoNode: ...

class _Plugin_bm3d_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "bm3d" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Basic(
        self,
        ref: VideoNode | None = None,
        profile: _DataType | None = None,
        sigma: _SingleAndSequence[float] | None = None,
        block_size: int | None = None,
        block_step: int | None = None,
        group_size: int | None = None,
        bm_range: int | None = None,
        bm_step: int | None = None,
        th_mse: float | None = None,
        hard_thr: float | None = None,
        matrix: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Final(
        self,
        ref: VideoNode,
        profile: _DataType | None = None,
        sigma: _SingleAndSequence[float] | None = None,
        block_size: int | None = None,
        block_step: int | None = None,
        group_size: int | None = None,
        bm_range: int | None = None,
        bm_step: int | None = None,
        th_mse: float | None = None,
        matrix: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def OPP2RGB(self, sample: int | None = None) -> ConstantFormatVideoNode: ...
    def RGB2OPP(self, sample: int | None = None) -> ConstantFormatVideoNode: ...
    def VAggregate(self, radius: int | None = None, sample: int | None = None) -> ConstantFormatVideoNode: ...
    def VBasic(
        self,
        ref: VideoNode | None = None,
        profile: _DataType | None = None,
        sigma: _SingleAndSequence[float] | None = None,
        radius: int | None = None,
        block_size: int | None = None,
        block_step: int | None = None,
        group_size: int | None = None,
        bm_range: int | None = None,
        bm_step: int | None = None,
        ps_num: int | None = None,
        ps_range: int | None = None,
        ps_step: int | None = None,
        th_mse: float | None = None,
        hard_thr: float | None = None,
        matrix: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def VFinal(
        self,
        ref: VideoNode,
        profile: _DataType | None = None,
        sigma: _SingleAndSequence[float] | None = None,
        radius: int | None = None,
        block_size: int | None = None,
        block_step: int | None = None,
        group_size: int | None = None,
        bm_range: int | None = None,
        bm_step: int | None = None,
        ps_num: int | None = None,
        ps_range: int | None = None,
        ps_step: int | None = None,
        th_mse: float | None = None,
        matrix: int | None = None,
    ) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: bilateralgpu

class _Plugin_bilateralgpu_Core_Bound(Plugin):
    """This class implements the module definitions for the "akarin" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Bilateral(
        self,
        clip: VideoNode,
        sigma_spatial: _SingleAndSequence[float] | None = None,
        sigma_color: _SingleAndSequence[float] | None = None,
        radius: _SingleAndSequence[int] | None = None,
        device_id: int | None = None,
        num_streams: int | None = None,
        use_shared_memory: int | None = None,
        ref: VideoNode | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Version(self) -> bytes: ...

class _Plugin_bilateralgpu_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "akarin" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Bilateral(
        self,
        sigma_spatial: _SingleAndSequence[float] | None = None,
        sigma_color: _SingleAndSequence[float] | None = None,
        radius: _SingleAndSequence[int] | None = None,
        device_id: int | None = None,
        num_streams: int | None = None,
        use_shared_memory: int | None = None,
        ref: VideoNode | None = None,
    ) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: bilateralgpu_rtc

class _Plugin_bilateralgpu_rtc_Core_Bound(Plugin):
    """This class implements the module definitions for the "akarin" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Bilateral(
        self,
        clip: VideoNode,
        sigma_spatial: _SingleAndSequence[float] | None = None,
        sigma_color: _SingleAndSequence[float] | None = None,
        radius: _SingleAndSequence[int] | None = None,
        device_id: int | None = None,
        num_streams: int | None = None,
        use_shared_memory: int | None = None,
        block_x: int | None = None,
        block_y: int | None = None,
        ref: VideoNode | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Version(self) -> bytes: ...

class _Plugin_bilateralgpu_rtc_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "akarin" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Bilateral(
        self,
        sigma_spatial: _SingleAndSequence[float] | None = None,
        sigma_color: _SingleAndSequence[float] | None = None,
        radius: _SingleAndSequence[int] | None = None,
        device_id: int | None = None,
        num_streams: int | None = None,
        use_shared_memory: int | None = None,
        block_x: int | None = None,
        block_y: int | None = None,
        ref: VideoNode | None = None,
    ) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: bm3dcpu

class _Plugin_bm3dcpu_Core_Bound(Plugin):
    """This class implements the module definitions for the "bm3dcpu" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def BM3D(
        self,
        clip: VideoNode,
        ref: VideoNode | None = None,
        sigma: _SingleAndSequence[float] | None = None,
        block_step: _SingleAndSequence[int] | None = None,
        bm_range: _SingleAndSequence[int] | None = None,
        radius: int | None = None,
        ps_num: int | None = None,
        ps_range: int | None = None,
        chroma: int | None = None,
        zero_init: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def BM3Dv2(
        self,
        clip: VideoNode,
        ref: VideoNode | None = None,
        sigma: _SingleAndSequence[float] | None = None,
        block_step: _SingleAndSequence[int] | None = None,
        bm_range: _SingleAndSequence[int] | None = None,
        radius: int | None = None,
        ps_num: int | None = None,
        ps_range: int | None = None,
        chroma: int | None = None,
        zero_init: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def VAggregate(
        self, clip: VideoNode, src: VideoNode, planes: _SingleAndSequence[int]
    ) -> ConstantFormatVideoNode: ...

class _Plugin_bm3dcpu_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "bm3dcpu" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def BM3D(
        self,
        ref: VideoNode | None = None,
        sigma: _SingleAndSequence[float] | None = None,
        block_step: _SingleAndSequence[int] | None = None,
        bm_range: _SingleAndSequence[int] | None = None,
        radius: int | None = None,
        ps_num: int | None = None,
        ps_range: int | None = None,
        chroma: int | None = None,
        zero_init: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def BM3Dv2(
        self,
        ref: VideoNode | None = None,
        sigma: _SingleAndSequence[float] | None = None,
        block_step: _SingleAndSequence[int] | None = None,
        bm_range: _SingleAndSequence[int] | None = None,
        radius: int | None = None,
        ps_num: int | None = None,
        ps_range: int | None = None,
        chroma: int | None = None,
        zero_init: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def VAggregate(self, src: VideoNode, planes: _SingleAndSequence[int]) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: bm3dcuda

class _Plugin_bm3dcuda_Core_Bound(Plugin):
    """This class implements the module definitions for the "bm3dcuda" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def BM3D(
        self,
        clip: VideoNode,
        ref: VideoNode | None = None,
        sigma: _SingleAndSequence[float] | None = None,
        block_step: _SingleAndSequence[int] | None = None,
        bm_range: _SingleAndSequence[int] | None = None,
        radius: int | None = None,
        ps_num: _SingleAndSequence[int] | None = None,
        ps_range: _SingleAndSequence[int] | None = None,
        chroma: int | None = None,
        device_id: int | None = None,
        fast: int | None = None,
        extractor_exp: int | None = None,
        zero_init: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def BM3Dv2(
        self,
        clip: VideoNode,
        ref: VideoNode | None = None,
        sigma: _SingleAndSequence[float] | None = None,
        block_step: _SingleAndSequence[int] | None = None,
        bm_range: _SingleAndSequence[int] | None = None,
        radius: int | None = None,
        ps_num: _SingleAndSequence[int] | None = None,
        ps_range: _SingleAndSequence[int] | None = None,
        chroma: int | None = None,
        device_id: int | None = None,
        fast: int | None = None,
        extractor_exp: int | None = None,
        zero_init: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def VAggregate(
        self, clip: VideoNode, src: VideoNode, planes: _SingleAndSequence[int]
    ) -> ConstantFormatVideoNode: ...

class _Plugin_bm3dcuda_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "bm3dcuda" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def BM3D(
        self,
        ref: VideoNode | None = None,
        sigma: _SingleAndSequence[float] | None = None,
        block_step: _SingleAndSequence[int] | None = None,
        bm_range: _SingleAndSequence[int] | None = None,
        radius: int | None = None,
        ps_num: _SingleAndSequence[int] | None = None,
        ps_range: _SingleAndSequence[int] | None = None,
        chroma: int | None = None,
        device_id: int | None = None,
        fast: int | None = None,
        extractor_exp: int | None = None,
        zero_init: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def BM3Dv2(
        self,
        ref: VideoNode | None = None,
        sigma: _SingleAndSequence[float] | None = None,
        block_step: _SingleAndSequence[int] | None = None,
        bm_range: _SingleAndSequence[int] | None = None,
        radius: int | None = None,
        ps_num: _SingleAndSequence[int] | None = None,
        ps_range: _SingleAndSequence[int] | None = None,
        chroma: int | None = None,
        device_id: int | None = None,
        fast: int | None = None,
        extractor_exp: int | None = None,
        zero_init: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def VAggregate(self, src: VideoNode, planes: _SingleAndSequence[int]) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: bm3dcuda_rtc

class _Plugin_bm3dcuda_rtc_Core_Bound(Plugin):
    """This class implements the module definitions for the "bm3dcuda_rtc" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def BM3D(
        self,
        clip: VideoNode,
        ref: VideoNode | None = None,
        sigma: _SingleAndSequence[float] | None = None,
        block_step: _SingleAndSequence[int] | None = None,
        bm_range: _SingleAndSequence[int] | None = None,
        radius: int | None = None,
        ps_num: _SingleAndSequence[int] | None = None,
        ps_range: _SingleAndSequence[int] | None = None,
        chroma: int | None = None,
        device_id: int | None = None,
        fast: int | None = None,
        extractor_exp: int | None = None,
        bm_error_s: _SingleAndSequence[_DataType] | None = None,
        transform_2d_s: _SingleAndSequence[_DataType] | None = None,
        transform_1d_s: _SingleAndSequence[_DataType] | None = None,
        zero_init: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def BM3Dv2(
        self,
        clip: VideoNode,
        ref: VideoNode | None = None,
        sigma: _SingleAndSequence[float] | None = None,
        block_step: _SingleAndSequence[int] | None = None,
        bm_range: _SingleAndSequence[int] | None = None,
        radius: int | None = None,
        ps_num: _SingleAndSequence[int] | None = None,
        ps_range: _SingleAndSequence[int] | None = None,
        chroma: int | None = None,
        device_id: int | None = None,
        fast: int | None = None,
        extractor_exp: int | None = None,
        bm_error_s: _SingleAndSequence[_DataType] | None = None,
        transform_2d_s: _SingleAndSequence[_DataType] | None = None,
        transform_1d_s: _SingleAndSequence[_DataType] | None = None,
        zero_init: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def VAggregate(
        self, clip: VideoNode, src: VideoNode, planes: _SingleAndSequence[int]
    ) -> ConstantFormatVideoNode: ...

class _Plugin_bm3dcuda_rtc_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "bm3dcuda_rtc" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def BM3D(
        self,
        ref: VideoNode | None = None,
        sigma: _SingleAndSequence[float] | None = None,
        block_step: _SingleAndSequence[int] | None = None,
        bm_range: _SingleAndSequence[int] | None = None,
        radius: int | None = None,
        ps_num: _SingleAndSequence[int] | None = None,
        ps_range: _SingleAndSequence[int] | None = None,
        chroma: int | None = None,
        device_id: int | None = None,
        fast: int | None = None,
        extractor_exp: int | None = None,
        bm_error_s: _SingleAndSequence[_DataType] | None = None,
        transform_2d_s: _SingleAndSequence[_DataType] | None = None,
        transform_1d_s: _SingleAndSequence[_DataType] | None = None,
        zero_init: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def BM3Dv2(
        self,
        ref: VideoNode | None = None,
        sigma: _SingleAndSequence[float] | None = None,
        block_step: _SingleAndSequence[int] | None = None,
        bm_range: _SingleAndSequence[int] | None = None,
        radius: int | None = None,
        ps_num: _SingleAndSequence[int] | None = None,
        ps_range: _SingleAndSequence[int] | None = None,
        chroma: int | None = None,
        device_id: int | None = None,
        fast: int | None = None,
        extractor_exp: int | None = None,
        bm_error_s: _SingleAndSequence[_DataType] | None = None,
        transform_2d_s: _SingleAndSequence[_DataType] | None = None,
        transform_1d_s: _SingleAndSequence[_DataType] | None = None,
        zero_init: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def VAggregate(self, src: VideoNode, planes: _SingleAndSequence[int]) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: bm3dhip

class _Plugin_bm3dhip_Core_Bound(Plugin):
    """This class implements the module definitions for the "bm3dhip" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def BM3D(
        self,
        clip: VideoNode,
        ref: VideoNode | None = None,
        sigma: _SingleAndSequence[float] | None = None,
        block_step: _SingleAndSequence[int] | None = None,
        bm_range: _SingleAndSequence[int] | None = None,
        radius: int | None = None,
        ps_num: _SingleAndSequence[int] | None = None,
        ps_range: _SingleAndSequence[int] | None = None,
        chroma: int | None = None,
        device_id: int | None = None,
        fast: int | None = None,
        extractor_exp: int | None = None,
        zero_init: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def BM3Dv2(
        self,
        clip: VideoNode,
        ref: VideoNode | None = None,
        sigma: _SingleAndSequence[float] | None = None,
        block_step: _SingleAndSequence[int] | None = None,
        bm_range: _SingleAndSequence[int] | None = None,
        radius: int | None = None,
        ps_num: _SingleAndSequence[int] | None = None,
        ps_range: _SingleAndSequence[int] | None = None,
        chroma: int | None = None,
        device_id: int | None = None,
        fast: int | None = None,
        extractor_exp: int | None = None,
        zero_init: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def VAggregate(
        self, clip: VideoNode, src: VideoNode, planes: _SingleAndSequence[int]
    ) -> ConstantFormatVideoNode: ...

class _Plugin_bm3dhip_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "bm3dhip" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def BM3D(
        self,
        ref: VideoNode | None = None,
        sigma: _SingleAndSequence[float] | None = None,
        block_step: _SingleAndSequence[int] | None = None,
        bm_range: _SingleAndSequence[int] | None = None,
        radius: int | None = None,
        ps_num: _SingleAndSequence[int] | None = None,
        ps_range: _SingleAndSequence[int] | None = None,
        chroma: int | None = None,
        device_id: int | None = None,
        fast: int | None = None,
        extractor_exp: int | None = None,
        zero_init: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def BM3Dv2(
        self,
        ref: VideoNode | None = None,
        sigma: _SingleAndSequence[float] | None = None,
        block_step: _SingleAndSequence[int] | None = None,
        bm_range: _SingleAndSequence[int] | None = None,
        radius: int | None = None,
        ps_num: _SingleAndSequence[int] | None = None,
        ps_range: _SingleAndSequence[int] | None = None,
        chroma: int | None = None,
        device_id: int | None = None,
        fast: int | None = None,
        extractor_exp: int | None = None,
        zero_init: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def VAggregate(self, src: VideoNode, planes: _SingleAndSequence[int]) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: bm3dsycl

class _Plugin_bm3dsycl_Core_Bound(Plugin):
    """This class implements the module definitions for the "bm3dsycl" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def BM3D(
        self,
        clip: VideoNode,
        ref: VideoNode | None = None,
        sigma: _SingleAndSequence[float] | None = None,
        block_step: _SingleAndSequence[int] | None = None,
        bm_range: _SingleAndSequence[int] | None = None,
        radius: int | None = None,
        ps_num: _SingleAndSequence[int] | None = None,
        ps_range: _SingleAndSequence[int] | None = None,
        chroma: int | None = None,
        device_id: int | None = None,
        fast: int | None = None,
        extractor_exp: int | None = None,
        zero_init: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def BM3Dv2(
        self,
        clip: VideoNode,
        ref: VideoNode | None = None,
        sigma: _SingleAndSequence[float] | None = None,
        block_step: _SingleAndSequence[int] | None = None,
        bm_range: _SingleAndSequence[int] | None = None,
        radius: int | None = None,
        ps_num: _SingleAndSequence[int] | None = None,
        ps_range: _SingleAndSequence[int] | None = None,
        chroma: int | None = None,
        device_id: int | None = None,
        fast: int | None = None,
        extractor_exp: int | None = None,
        zero_init: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def VAggregate(
        self, clip: VideoNode, src: VideoNode, planes: _SingleAndSequence[int]
    ) -> ConstantFormatVideoNode: ...

class _Plugin_bm3dsycl_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "bm3dsycl" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def BM3D(
        self,
        ref: VideoNode | None = None,
        sigma: _SingleAndSequence[float] | None = None,
        block_step: _SingleAndSequence[int] | None = None,
        bm_range: _SingleAndSequence[int] | None = None,
        radius: int | None = None,
        ps_num: _SingleAndSequence[int] | None = None,
        ps_range: _SingleAndSequence[int] | None = None,
        chroma: int | None = None,
        device_id: int | None = None,
        fast: int | None = None,
        extractor_exp: int | None = None,
        zero_init: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def BM3Dv2(
        self,
        ref: VideoNode | None = None,
        sigma: _SingleAndSequence[float] | None = None,
        block_step: _SingleAndSequence[int] | None = None,
        bm_range: _SingleAndSequence[int] | None = None,
        radius: int | None = None,
        ps_num: _SingleAndSequence[int] | None = None,
        ps_range: _SingleAndSequence[int] | None = None,
        chroma: int | None = None,
        device_id: int | None = None,
        fast: int | None = None,
        extractor_exp: int | None = None,
        zero_init: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def VAggregate(self, src: VideoNode, planes: _SingleAndSequence[int]) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: bs

_ReturnDict_bs_TrackInfo = TypedDict(
    "_ReturnDict_bs_TrackInfo",
    {
        "mediatype": int,
        "mediatypestr": _DataType,
        "codec": int,
        "codecstr": _DataType,
        "disposition": int,
        "dispositionstr": _DataType,
    },
)

class _Plugin_bs_Core_Bound(Plugin):
    """This class implements the module definitions for the "bs" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def AudioSource(
        self,
        source: _DataType,
        track: int | None = None,
        adjustdelay: int | None = None,
        threads: int | None = None,
        enable_drefs: int | None = None,
        use_absolute_path: int | None = None,
        drc_scale: float | None = None,
        cachemode: int | None = None,
        cachepath: _DataType | None = None,
        cachesize: int | None = None,
        showprogress: int | None = None,
    ) -> AudioNode: ...
    def Metadata(
        self,
        source: _DataType,
        track: int | None = None,
        enable_drefs: int | None = None,
        use_absolute_path: int | None = None,
    ) -> dict[str, Any]: ...
    def SetDebugOutput(self, enable: int) -> None: ...
    def SetFFmpegLogLevel(self, level: int) -> int: ...
    def TrackInfo(
        self, source: _DataType, enable_drefs: int | None = None, use_absolute_path: int | None = None
    ) -> _ReturnDict_bs_TrackInfo: ...
    def VideoSource(
        self,
        source: _DataType,
        track: int | None = None,
        variableformat: int | None = None,
        fpsnum: int | None = None,
        fpsden: int | None = None,
        rff: int | None = None,
        threads: int | None = None,
        seekpreroll: int | None = None,
        enable_drefs: int | None = None,
        use_absolute_path: int | None = None,
        cachemode: int | None = None,
        cachepath: _DataType | None = None,
        cachesize: int | None = None,
        hwdevice: _DataType | None = None,
        extrahwframes: int | None = None,
        timecodes: _DataType | None = None,
        start_number: int | None = None,
        viewid: int | None = None,
        showprogress: int | None = None,
    ) -> VideoNode: ...

# end implementation

# implementation: bwdif

class _Plugin_bwdif_Core_Bound(Plugin):
    """This class implements the module definitions for the "bwdif" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Bwdif(
        self, clip: VideoNode, field: int, edeint: VideoNode | None = None, opt: int | None = None
    ) -> ConstantFormatVideoNode: ...

class _Plugin_bwdif_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "bwdif" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Bwdif(self, field: int, edeint: VideoNode | None = None, opt: int | None = None) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: cs

class _Plugin_cs_Core_Bound(Plugin):
    """This class implements the module definitions for the "cs" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def ConvertColor(
        self,
        clip: VideoNode,
        output_profile: _DataType,
        input_profile: _DataType | None = None,
        float_output: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def ImageSource(
        self,
        source: _DataType,
        subsampling_pad: int | None = None,
        jpeg_rgb: int | None = None,
        jpeg_fancy_upsampling: int | None = None,
        jpeg_cmyk_profile: _DataType | None = None,
        jpeg_cmyk_target_profile: _DataType | None = None,
    ) -> ConstantFormatVideoNode: ...

class _Plugin_cs_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "cs" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def ConvertColor(
        self, output_profile: _DataType, input_profile: _DataType | None = None, float_output: int | None = None
    ) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: d2v

class _Plugin_d2v_Core_Bound(Plugin):
    """This class implements the module definitions for the "d2v" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Source(
        self, input: _DataType, threads: int | None = None, nocrop: int | None = None, rff: int | None = None
    ) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: dctf

class _Plugin_dctf_Core_Bound(Plugin):
    """This class implements the module definitions for the "dctf" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def DCTFilter(
        self, clip: VideoNode, factors: _SingleAndSequence[float], planes: _SingleAndSequence[int] | None = None
    ) -> ConstantFormatVideoNode: ...

class _Plugin_dctf_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "dctf" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def DCTFilter(
        self, factors: _SingleAndSequence[float], planes: _SingleAndSequence[int] | None = None
    ) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: deblock

class _Plugin_deblock_Core_Bound(Plugin):
    """This class implements the module definitions for the "deblock" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Deblock(
        self,
        clip: VideoNode,
        quant: int | None = None,
        aoffset: int | None = None,
        boffset: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...

class _Plugin_deblock_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "deblock" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Deblock(
        self,
        quant: int | None = None,
        aoffset: int | None = None,
        boffset: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...

# implementation: descale

class _Plugin_descale_Core_Bound(Plugin):
    """This class implements the module definitions for the "descale" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Bicubic(
        self,
        src: VideoNode,
        width: int,
        height: int,
        b: float | None = None,
        c: float | None = None,
        blur: float | None = None,
        post_conv: _SingleAndSequence[float] | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        border_handling: int | None = None,
        ignore_mask: VideoNode | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Bilinear(
        self,
        src: VideoNode,
        width: int,
        height: int,
        blur: float | None = None,
        post_conv: _SingleAndSequence[float] | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        border_handling: int | None = None,
        ignore_mask: VideoNode | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Debicubic(
        self,
        src: VideoNode,
        width: int,
        height: int,
        b: float | None = None,
        c: float | None = None,
        blur: float | None = None,
        post_conv: _SingleAndSequence[float] | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        border_handling: int | None = None,
        ignore_mask: VideoNode | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Debilinear(
        self,
        src: VideoNode,
        width: int,
        height: int,
        blur: float | None = None,
        post_conv: _SingleAndSequence[float] | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        border_handling: int | None = None,
        ignore_mask: VideoNode | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Decustom(
        self,
        src: VideoNode,
        width: int,
        height: int,
        custom_kernel: _VSMapValueCallback[_VapourSynthMapValue],
        taps: int,
        blur: float | None = None,
        post_conv: _SingleAndSequence[float] | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        border_handling: int | None = None,
        ignore_mask: VideoNode | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Delanczos(
        self,
        src: VideoNode,
        width: int,
        height: int,
        taps: int | None = None,
        blur: float | None = None,
        post_conv: _SingleAndSequence[float] | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        border_handling: int | None = None,
        ignore_mask: VideoNode | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Depoint(
        self,
        src: VideoNode,
        width: int,
        height: int,
        blur: float | None = None,
        post_conv: _SingleAndSequence[float] | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        border_handling: int | None = None,
        ignore_mask: VideoNode | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Despline16(
        self,
        src: VideoNode,
        width: int,
        height: int,
        blur: float | None = None,
        post_conv: _SingleAndSequence[float] | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        border_handling: int | None = None,
        ignore_mask: VideoNode | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Despline36(
        self,
        src: VideoNode,
        width: int,
        height: int,
        blur: float | None = None,
        post_conv: _SingleAndSequence[float] | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        border_handling: int | None = None,
        ignore_mask: VideoNode | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Despline64(
        self,
        src: VideoNode,
        width: int,
        height: int,
        blur: float | None = None,
        post_conv: _SingleAndSequence[float] | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        border_handling: int | None = None,
        ignore_mask: VideoNode | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Lanczos(
        self,
        src: VideoNode,
        width: int,
        height: int,
        taps: int | None = None,
        blur: float | None = None,
        post_conv: _SingleAndSequence[float] | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        border_handling: int | None = None,
        ignore_mask: VideoNode | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Point(
        self,
        src: VideoNode,
        width: int,
        height: int,
        blur: float | None = None,
        post_conv: _SingleAndSequence[float] | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        border_handling: int | None = None,
        ignore_mask: VideoNode | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def ScaleCustom(
        self,
        src: VideoNode,
        width: int,
        height: int,
        custom_kernel: _VSMapValueCallback[_VapourSynthMapValue],
        taps: int,
        blur: float | None = None,
        post_conv: _SingleAndSequence[float] | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        border_handling: int | None = None,
        ignore_mask: VideoNode | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Spline16(
        self,
        src: VideoNode,
        width: int,
        height: int,
        blur: float | None = None,
        post_conv: _SingleAndSequence[float] | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        border_handling: int | None = None,
        ignore_mask: VideoNode | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Spline36(
        self,
        src: VideoNode,
        width: int,
        height: int,
        blur: float | None = None,
        post_conv: _SingleAndSequence[float] | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        border_handling: int | None = None,
        ignore_mask: VideoNode | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Spline64(
        self,
        src: VideoNode,
        width: int,
        height: int,
        blur: float | None = None,
        post_conv: _SingleAndSequence[float] | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        border_handling: int | None = None,
        ignore_mask: VideoNode | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...

class _Plugin_descale_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "descale" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Bicubic(
        self,
        width: int,
        height: int,
        b: float | None = None,
        c: float | None = None,
        blur: float | None = None,
        post_conv: _SingleAndSequence[float] | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        border_handling: int | None = None,
        ignore_mask: VideoNode | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Bilinear(
        self,
        width: int,
        height: int,
        blur: float | None = None,
        post_conv: _SingleAndSequence[float] | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        border_handling: int | None = None,
        ignore_mask: VideoNode | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Debicubic(
        self,
        width: int,
        height: int,
        b: float | None = None,
        c: float | None = None,
        blur: float | None = None,
        post_conv: _SingleAndSequence[float] | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        border_handling: int | None = None,
        ignore_mask: VideoNode | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Debilinear(
        self,
        width: int,
        height: int,
        blur: float | None = None,
        post_conv: _SingleAndSequence[float] | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        border_handling: int | None = None,
        ignore_mask: VideoNode | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Decustom(
        self,
        width: int,
        height: int,
        custom_kernel: _VSMapValueCallback[_VapourSynthMapValue],
        taps: int,
        blur: float | None = None,
        post_conv: _SingleAndSequence[float] | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        border_handling: int | None = None,
        ignore_mask: VideoNode | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Delanczos(
        self,
        width: int,
        height: int,
        taps: int | None = None,
        blur: float | None = None,
        post_conv: _SingleAndSequence[float] | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        border_handling: int | None = None,
        ignore_mask: VideoNode | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Depoint(
        self,
        width: int,
        height: int,
        blur: float | None = None,
        post_conv: _SingleAndSequence[float] | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        border_handling: int | None = None,
        ignore_mask: VideoNode | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Despline16(
        self,
        width: int,
        height: int,
        blur: float | None = None,
        post_conv: _SingleAndSequence[float] | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        border_handling: int | None = None,
        ignore_mask: VideoNode | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Despline36(
        self,
        width: int,
        height: int,
        blur: float | None = None,
        post_conv: _SingleAndSequence[float] | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        border_handling: int | None = None,
        ignore_mask: VideoNode | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Despline64(
        self,
        width: int,
        height: int,
        blur: float | None = None,
        post_conv: _SingleAndSequence[float] | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        border_handling: int | None = None,
        ignore_mask: VideoNode | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Lanczos(
        self,
        width: int,
        height: int,
        taps: int | None = None,
        blur: float | None = None,
        post_conv: _SingleAndSequence[float] | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        border_handling: int | None = None,
        ignore_mask: VideoNode | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Point(
        self,
        width: int,
        height: int,
        blur: float | None = None,
        post_conv: _SingleAndSequence[float] | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        border_handling: int | None = None,
        ignore_mask: VideoNode | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def ScaleCustom(
        self,
        width: int,
        height: int,
        custom_kernel: _VSMapValueCallback[_VapourSynthMapValue],
        taps: int,
        blur: float | None = None,
        post_conv: _SingleAndSequence[float] | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        border_handling: int | None = None,
        ignore_mask: VideoNode | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Spline16(
        self,
        width: int,
        height: int,
        blur: float | None = None,
        post_conv: _SingleAndSequence[float] | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        border_handling: int | None = None,
        ignore_mask: VideoNode | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Spline36(
        self,
        width: int,
        height: int,
        blur: float | None = None,
        post_conv: _SingleAndSequence[float] | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        border_handling: int | None = None,
        ignore_mask: VideoNode | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Spline64(
        self,
        width: int,
        height: int,
        blur: float | None = None,
        post_conv: _SingleAndSequence[float] | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        border_handling: int | None = None,
        ignore_mask: VideoNode | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: dfttest

class _Plugin_dfttest_Core_Bound(Plugin):
    """This class implements the module definitions for the "dfttest" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def DFTTest(
        self,
        clip: VideoNode,
        ftype: int | None = None,
        sigma: float | None = None,
        sigma2: float | None = None,
        pmin: float | None = None,
        pmax: float | None = None,
        sbsize: int | None = None,
        smode: int | None = None,
        sosize: int | None = None,
        tbsize: int | None = None,
        tmode: int | None = None,
        tosize: int | None = None,
        swin: int | None = None,
        twin: int | None = None,
        sbeta: float | None = None,
        tbeta: float | None = None,
        zmean: int | None = None,
        f0beta: float | None = None,
        nlocation: _SingleAndSequence[int] | None = None,
        alpha: float | None = None,
        slocation: _SingleAndSequence[float] | None = None,
        ssx: _SingleAndSequence[float] | None = None,
        ssy: _SingleAndSequence[float] | None = None,
        sst: _SingleAndSequence[float] | None = None,
        ssystem: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...

class _Plugin_dfttest_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "dfttest" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def DFTTest(
        self,
        ftype: int | None = None,
        sigma: float | None = None,
        sigma2: float | None = None,
        pmin: float | None = None,
        pmax: float | None = None,
        sbsize: int | None = None,
        smode: int | None = None,
        sosize: int | None = None,
        tbsize: int | None = None,
        tmode: int | None = None,
        tosize: int | None = None,
        swin: int | None = None,
        twin: int | None = None,
        sbeta: float | None = None,
        tbeta: float | None = None,
        zmean: int | None = None,
        f0beta: float | None = None,
        nlocation: _SingleAndSequence[int] | None = None,
        alpha: float | None = None,
        slocation: _SingleAndSequence[float] | None = None,
        ssx: _SingleAndSequence[float] | None = None,
        ssy: _SingleAndSequence[float] | None = None,
        sst: _SingleAndSequence[float] | None = None,
        ssystem: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: dfttest2_cpu

_ReturnDict_dfttest2_cpu_Version = TypedDict(
    "_ReturnDict_dfttest2_cpu_Version",
    {
        "version": bytes,
        "dispatch_targets": list[bytes],
    },
)

class _Plugin_dfttest2_cpu_Core_Bound(Plugin):
    """This class implements the module definitions for the "dfttest2_cpu" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def DFTTest(
        self,
        clip: VideoNode,
        window: _SingleAndSequence[float],
        sigma: _SingleAndSequence[float],
        sigma2: float,
        pmin: float,
        pmax: float,
        filter_type: int,
        radius: int | None = None,
        block_size: int | None = None,
        block_step: int | None = None,
        zero_mean: int | None = None,
        window_freq: _SingleAndSequence[float] | None = None,
        planes: _SingleAndSequence[int] | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def RDFT(self, data: _SingleAndSequence[float], shape: _SingleAndSequence[int]) -> ConstantFormatVideoNode: ...
    def Version(self) -> _ReturnDict_dfttest2_cpu_Version: ...

class _Plugin_dfttest2_cpu_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "dfttest2_cpu" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def DFTTest(
        self,
        window: _SingleAndSequence[float],
        sigma: _SingleAndSequence[float],
        sigma2: float,
        pmin: float,
        pmax: float,
        filter_type: int,
        radius: int | None = None,
        block_size: int | None = None,
        block_step: int | None = None,
        zero_mean: int | None = None,
        window_freq: _SingleAndSequence[float] | None = None,
        planes: _SingleAndSequence[int] | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: dfttest2_cuda

_ReturnDict_dfttest2_cuda_Version = TypedDict(
    "_ReturnDict_dfttest2_cuda_Version",
    {
        "version": bytes,
        "cufft_version": int,
        "cufft_version_build": int,
    },
)

class _Plugin_dfttest2_cuda_Core_Bound(Plugin):
    """This class implements the module definitions for the "dfttest2_cuda" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def DFTTest(
        self,
        clip: VideoNode,
        kernel: _SingleAndSequence[_DataType],
        radius: int | None = None,
        block_size: int | None = None,
        block_step: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
        in_place: int | None = None,
        device_id: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def RDFT(self, data: _SingleAndSequence[float], shape: _SingleAndSequence[int]) -> ConstantFormatVideoNode: ...
    def ToSingle(self, data: _SingleAndSequence[float]) -> ConstantFormatVideoNode: ...
    def Version(self) -> _ReturnDict_dfttest2_cuda_Version: ...

class _Plugin_dfttest2_cuda_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "dfttest2_cuda" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def DFTTest(
        self,
        kernel: _SingleAndSequence[_DataType],
        radius: int | None = None,
        block_size: int | None = None,
        block_step: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
        in_place: int | None = None,
        device_id: int | None = None,
    ) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: dfttest2_nvrtc

class _Plugin_dfttest2_nvrtc_Core_Bound(Plugin):
    """This class implements the module definitions for the "dfttest2_nvrtc" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def DFTTest(
        self,
        clip: VideoNode,
        kernel: _SingleAndSequence[_DataType],
        radius: int | None = None,
        block_size: int | None = None,
        block_step: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
        in_place: int | None = None,
        device_id: int | None = None,
        num_streams: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def RDFT(self, data: _SingleAndSequence[float], shape: _SingleAndSequence[int]) -> ConstantFormatVideoNode: ...
    def ToSingle(self, data: _SingleAndSequence[float]) -> ConstantFormatVideoNode: ...
    def Version(self) -> bytes: ...

class _Plugin_dfttest2_nvrtc_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "dfttest2_nvrtc" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def DFTTest(
        self,
        kernel: _SingleAndSequence[_DataType],
        radius: int | None = None,
        block_size: int | None = None,
        block_step: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
        in_place: int | None = None,
        device_id: int | None = None,
        num_streams: int | None = None,
    ) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: dgdecodenv

class _Plugin_dgdecodenv_Core_Bound(Plugin):
    """This class implements the module definitions for the "dgdecodenv" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def DGSource(
        self,
        source: _DataType,
        i420: int | None = None,
        deinterlace: int | None = None,
        use_top_field: int | None = None,
        use_pf: int | None = None,
        ct: int | None = None,
        cb: int | None = None,
        cl: int | None = None,
        cr: int | None = None,
        rw: int | None = None,
        rh: int | None = None,
        fieldop: int | None = None,
        show: int | None = None,
        show2: _DataType | None = None,
    ) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: dvdsrc2

class _Plugin_dvdsrc2_Core_Bound(Plugin):
    """This class implements the module definitions for the "dvdsrc2" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def FullVts(
        self, path: _DataType, vts: int, ranges: _SingleAndSequence[int] | None = None
    ) -> ConstantFormatVideoNode: ...
    def FullVtsAc3(
        self, path: _DataType, vts: int, audio: int, ranges: _SingleAndSequence[int] | None = None
    ) -> AudioNode: ...
    def FullVtsLpcm(
        self, path: _DataType, vts: int, audio: int, ranges: _SingleAndSequence[int] | None = None
    ) -> AudioNode: ...
    def Ifo(self, path: _DataType, ifo: int) -> _DataType: ...
    def RawAc3(
        self, path: _DataType, vts: int, audio: int, ranges: _SingleAndSequence[int] | None = None
    ) -> AudioNode: ...

# end implementation

# implementation: edgemasks

class _Plugin_edgemasks_Core_Bound(Plugin):
    """This class implements the module definitions for the "edgemasks" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Cross(
        self,
        clip: VideoNode,
        planes: _SingleAndSequence[int] | None = None,
        scale: _SingleAndSequence[float] | None = None,
        opt: int | None = None,
    ) -> VideoNode: ...
    def ExKirsch(
        self,
        clip: VideoNode,
        planes: _SingleAndSequence[int] | None = None,
        scale: _SingleAndSequence[float] | None = None,
        opt: int | None = None,
    ) -> VideoNode: ...
    def ExPrewitt(
        self,
        clip: VideoNode,
        planes: _SingleAndSequence[int] | None = None,
        scale: _SingleAndSequence[float] | None = None,
        opt: int | None = None,
    ) -> VideoNode: ...
    def ExSobel(
        self,
        clip: VideoNode,
        planes: _SingleAndSequence[int] | None = None,
        scale: _SingleAndSequence[float] | None = None,
        opt: int | None = None,
    ) -> VideoNode: ...
    def FDoG(
        self,
        clip: VideoNode,
        planes: _SingleAndSequence[int] | None = None,
        scale: _SingleAndSequence[float] | None = None,
        opt: int | None = None,
    ) -> VideoNode: ...
    def Kirsch(
        self,
        clip: VideoNode,
        planes: _SingleAndSequence[int] | None = None,
        scale: _SingleAndSequence[float] | None = None,
        opt: int | None = None,
    ) -> VideoNode: ...
    def Kroon(
        self,
        clip: VideoNode,
        planes: _SingleAndSequence[int] | None = None,
        scale: _SingleAndSequence[float] | None = None,
        opt: int | None = None,
    ) -> VideoNode: ...
    def Prewitt(
        self,
        clip: VideoNode,
        planes: _SingleAndSequence[int] | None = None,
        scale: _SingleAndSequence[float] | None = None,
        opt: int | None = None,
    ) -> VideoNode: ...
    def Robinson3(
        self,
        clip: VideoNode,
        planes: _SingleAndSequence[int] | None = None,
        scale: _SingleAndSequence[float] | None = None,
        opt: int | None = None,
    ) -> VideoNode: ...
    def Robinson5(
        self,
        clip: VideoNode,
        planes: _SingleAndSequence[int] | None = None,
        scale: _SingleAndSequence[float] | None = None,
        opt: int | None = None,
    ) -> VideoNode: ...
    def RScharr(
        self,
        clip: VideoNode,
        planes: _SingleAndSequence[int] | None = None,
        scale: _SingleAndSequence[float] | None = None,
        opt: int | None = None,
    ) -> VideoNode: ...
    def Scharr(
        self,
        clip: VideoNode,
        planes: _SingleAndSequence[int] | None = None,
        scale: _SingleAndSequence[float] | None = None,
        opt: int | None = None,
    ) -> VideoNode: ...
    def Sobel(
        self,
        clip: VideoNode,
        planes: _SingleAndSequence[int] | None = None,
        scale: _SingleAndSequence[float] | None = None,
        opt: int | None = None,
    ) -> VideoNode: ...
    def Tritical(
        self,
        clip: VideoNode,
        planes: _SingleAndSequence[int] | None = None,
        scale: _SingleAndSequence[float] | None = None,
        opt: int | None = None,
    ) -> VideoNode: ...

class _Plugin_edgemasks_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "edgemasks" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Cross(
        self,
        planes: _SingleAndSequence[int] | None = None,
        scale: _SingleAndSequence[float] | None = None,
        opt: int | None = None,
    ) -> VideoNode: ...
    def ExKirsch(
        self,
        planes: _SingleAndSequence[int] | None = None,
        scale: _SingleAndSequence[float] | None = None,
        opt: int | None = None,
    ) -> VideoNode: ...
    def ExPrewitt(
        self,
        planes: _SingleAndSequence[int] | None = None,
        scale: _SingleAndSequence[float] | None = None,
        opt: int | None = None,
    ) -> VideoNode: ...
    def ExSobel(
        self,
        planes: _SingleAndSequence[int] | None = None,
        scale: _SingleAndSequence[float] | None = None,
        opt: int | None = None,
    ) -> VideoNode: ...
    def FDoG(
        self,
        planes: _SingleAndSequence[int] | None = None,
        scale: _SingleAndSequence[float] | None = None,
        opt: int | None = None,
    ) -> VideoNode: ...
    def Kirsch(
        self,
        planes: _SingleAndSequence[int] | None = None,
        scale: _SingleAndSequence[float] | None = None,
        opt: int | None = None,
    ) -> VideoNode: ...
    def Kroon(
        self,
        planes: _SingleAndSequence[int] | None = None,
        scale: _SingleAndSequence[float] | None = None,
        opt: int | None = None,
    ) -> VideoNode: ...
    def Prewitt(
        self,
        planes: _SingleAndSequence[int] | None = None,
        scale: _SingleAndSequence[float] | None = None,
        opt: int | None = None,
    ) -> VideoNode: ...
    def Robinson3(
        self,
        planes: _SingleAndSequence[int] | None = None,
        scale: _SingleAndSequence[float] | None = None,
        opt: int | None = None,
    ) -> VideoNode: ...
    def Robinson5(
        self,
        planes: _SingleAndSequence[int] | None = None,
        scale: _SingleAndSequence[float] | None = None,
        opt: int | None = None,
    ) -> VideoNode: ...
    def RScharr(
        self,
        planes: _SingleAndSequence[int] | None = None,
        scale: _SingleAndSequence[float] | None = None,
        opt: int | None = None,
    ) -> VideoNode: ...
    def Scharr(
        self,
        planes: _SingleAndSequence[int] | None = None,
        scale: _SingleAndSequence[float] | None = None,
        opt: int | None = None,
    ) -> VideoNode: ...
    def Sobel(
        self,
        planes: _SingleAndSequence[int] | None = None,
        scale: _SingleAndSequence[float] | None = None,
        opt: int | None = None,
    ) -> VideoNode: ...
    def Tritical(
        self,
        planes: _SingleAndSequence[int] | None = None,
        scale: _SingleAndSequence[float] | None = None,
        opt: int | None = None,
    ) -> VideoNode: ...

# end implementation

# implementation: eedi2

class _Plugin_eedi2_Core_Bound(Plugin):
    """This class implements the module definitions for the "eedi2" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def EEDI2(
        self,
        clip: VideoNode,
        field: int,
        mthresh: int | None = None,
        lthresh: int | None = None,
        vthresh: int | None = None,
        estr: int | None = None,
        dstr: int | None = None,
        maxd: int | None = None,
        map: int | None = None,
        nt: int | None = None,
        pp: int | None = None,
    ) -> ConstantFormatVideoNode: ...

class _Plugin_eedi2_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "eedi2" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def EEDI2(
        self,
        field: int,
        mthresh: int | None = None,
        lthresh: int | None = None,
        vthresh: int | None = None,
        estr: int | None = None,
        dstr: int | None = None,
        maxd: int | None = None,
        map: int | None = None,
        nt: int | None = None,
        pp: int | None = None,
    ) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: eedi2cuda

class _Plugin_eedi2cuda_Core_Bound(Plugin):
    """This class implements the module definitions for the "eedi2cuda" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def AA2(
        self,
        clip: VideoNode,
        mthresh: int | None = None,
        lthresh: int | None = None,
        vthresh: int | None = None,
        estr: int | None = None,
        dstr: int | None = None,
        maxd: int | None = None,
        map: int | None = None,
        nt: int | None = None,
        pp: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
        num_streams: int | None = None,
        device_id: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def BuildConfig(self) -> _DataType: ...
    def EEDI2(
        self,
        clip: VideoNode,
        field: int,
        mthresh: int | None = None,
        lthresh: int | None = None,
        vthresh: int | None = None,
        estr: int | None = None,
        dstr: int | None = None,
        maxd: int | None = None,
        map: int | None = None,
        nt: int | None = None,
        pp: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
        num_streams: int | None = None,
        device_id: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Enlarge2(
        self,
        clip: VideoNode,
        mthresh: int | None = None,
        lthresh: int | None = None,
        vthresh: int | None = None,
        estr: int | None = None,
        dstr: int | None = None,
        maxd: int | None = None,
        map: int | None = None,
        nt: int | None = None,
        pp: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
        num_streams: int | None = None,
        device_id: int | None = None,
    ) -> ConstantFormatVideoNode: ...

class _Plugin_eedi2cuda_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "eedi2cuda" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def AA2(
        self,
        mthresh: int | None = None,
        lthresh: int | None = None,
        vthresh: int | None = None,
        estr: int | None = None,
        dstr: int | None = None,
        maxd: int | None = None,
        map: int | None = None,
        nt: int | None = None,
        pp: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
        num_streams: int | None = None,
        device_id: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def EEDI2(
        self,
        field: int,
        mthresh: int | None = None,
        lthresh: int | None = None,
        vthresh: int | None = None,
        estr: int | None = None,
        dstr: int | None = None,
        maxd: int | None = None,
        map: int | None = None,
        nt: int | None = None,
        pp: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
        num_streams: int | None = None,
        device_id: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Enlarge2(
        self,
        mthresh: int | None = None,
        lthresh: int | None = None,
        vthresh: int | None = None,
        estr: int | None = None,
        dstr: int | None = None,
        maxd: int | None = None,
        map: int | None = None,
        nt: int | None = None,
        pp: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
        num_streams: int | None = None,
        device_id: int | None = None,
    ) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: eedi3m

class _Plugin_eedi3m_Core_Bound(Plugin):
    """This class implements the module definitions for the "eedi3m" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def EEDI3(
        self,
        clip: VideoNode,
        field: int,
        dh: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
        alpha: float | None = None,
        beta: float | None = None,
        gamma: float | None = None,
        nrad: int | None = None,
        mdis: int | None = None,
        hp: int | None = None,
        ucubic: int | None = None,
        cost3: int | None = None,
        vcheck: int | None = None,
        vthresh0: float | None = None,
        vthresh1: float | None = None,
        vthresh2: float | None = None,
        sclip: VideoNode | None = None,
        mclip: VideoNode | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def EEDI3CL(
        self,
        clip: VideoNode,
        field: int,
        dh: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
        alpha: float | None = None,
        beta: float | None = None,
        gamma: float | None = None,
        nrad: int | None = None,
        mdis: int | None = None,
        hp: int | None = None,
        ucubic: int | None = None,
        cost3: int | None = None,
        vcheck: int | None = None,
        vthresh0: float | None = None,
        vthresh1: float | None = None,
        vthresh2: float | None = None,
        sclip: VideoNode | None = None,
        opt: int | None = None,
        device: int | None = None,
        list_device: int | None = None,
        info: int | None = None,
    ) -> ConstantFormatVideoNode: ...

class _Plugin_eedi3m_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "eedi3m" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def EEDI3(
        self,
        field: int,
        dh: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
        alpha: float | None = None,
        beta: float | None = None,
        gamma: float | None = None,
        nrad: int | None = None,
        mdis: int | None = None,
        hp: int | None = None,
        ucubic: int | None = None,
        cost3: int | None = None,
        vcheck: int | None = None,
        vthresh0: float | None = None,
        vthresh1: float | None = None,
        vthresh2: float | None = None,
        sclip: VideoNode | None = None,
        mclip: VideoNode | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def EEDI3CL(
        self,
        field: int,
        dh: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
        alpha: float | None = None,
        beta: float | None = None,
        gamma: float | None = None,
        nrad: int | None = None,
        mdis: int | None = None,
        hp: int | None = None,
        ucubic: int | None = None,
        cost3: int | None = None,
        vcheck: int | None = None,
        vthresh0: float | None = None,
        vthresh1: float | None = None,
        vthresh2: float | None = None,
        sclip: VideoNode | None = None,
        opt: int | None = None,
        device: int | None = None,
        list_device: int | None = None,
        info: int | None = None,
    ) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: ffms2

class _Plugin_ffms2_Core_Bound(Plugin):
    """This class implements the module definitions for the "ffms2" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def GetLogLevel(self) -> int: ...
    def Index(
        self,
        source: _DataType,
        cachefile: _DataType | None = None,
        indextracks: _SingleAndSequence[int] | None = None,
        errorhandling: int | None = None,
        overwrite: int | None = None,
        enable_drefs: int | None = None,
        use_absolute_path: int | None = None,
    ) -> _DataType: ...
    def SetLogLevel(self, level: int) -> int: ...
    def Source(
        self,
        source: _DataType,
        track: int | None = None,
        cache: int | None = None,
        cachefile: _DataType | None = None,
        fpsnum: int | None = None,
        fpsden: int | None = None,
        threads: int | None = None,
        timecodes: _DataType | None = None,
        seekmode: int | None = None,
        width: int | None = None,
        height: int | None = None,
        resizer: _DataType | None = None,
        format: int | None = None,
        alpha: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Version(self) -> _DataType: ...

# end implementation

# implementation: fft3dfilter

class _Plugin_fft3dfilter_Core_Bound(Plugin):
    """This class implements the module definitions for the "fft3dfilter" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def FFT3DFilter(
        self,
        clip: VideoNode,
        sigma: float | None = None,
        beta: float | None = None,
        planes: _SingleAndSequence[int] | None = None,
        bw: int | None = None,
        bh: int | None = None,
        bt: int | None = None,
        ow: int | None = None,
        oh: int | None = None,
        kratio: float | None = None,
        sharpen: float | None = None,
        scutoff: float | None = None,
        svr: float | None = None,
        smin: float | None = None,
        smax: float | None = None,
        measure: int | None = None,
        interlaced: int | None = None,
        wintype: int | None = None,
        pframe: int | None = None,
        px: int | None = None,
        py: int | None = None,
        pshow: int | None = None,
        pcutoff: float | None = None,
        pfactor: float | None = None,
        sigma2: float | None = None,
        sigma3: float | None = None,
        sigma4: float | None = None,
        degrid: float | None = None,
        dehalo: float | None = None,
        hr: float | None = None,
        ht: float | None = None,
        ncpu: int | None = None,
    ) -> ConstantFormatVideoNode: ...

class _Plugin_fft3dfilter_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "fft3dfilter" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def FFT3DFilter(
        self,
        sigma: float | None = None,
        beta: float | None = None,
        planes: _SingleAndSequence[int] | None = None,
        bw: int | None = None,
        bh: int | None = None,
        bt: int | None = None,
        ow: int | None = None,
        oh: int | None = None,
        kratio: float | None = None,
        sharpen: float | None = None,
        scutoff: float | None = None,
        svr: float | None = None,
        smin: float | None = None,
        smax: float | None = None,
        measure: int | None = None,
        interlaced: int | None = None,
        wintype: int | None = None,
        pframe: int | None = None,
        px: int | None = None,
        py: int | None = None,
        pshow: int | None = None,
        pcutoff: float | None = None,
        pfactor: float | None = None,
        sigma2: float | None = None,
        sigma3: float | None = None,
        sigma4: float | None = None,
        degrid: float | None = None,
        dehalo: float | None = None,
        hr: float | None = None,
        ht: float | None = None,
        ncpu: int | None = None,
    ) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: fmtc

class _Plugin_fmtc_Core_Bound(Plugin):
    """This class implements the module definitions for the "fmtc" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def bitdepth(
        self,
        clip: VideoNode,
        csp: int | None = None,
        bits: int | None = None,
        flt: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
        fulls: int | None = None,
        fulld: int | None = None,
        dmode: int | None = None,
        ampo: float | None = None,
        ampn: float | None = None,
        dyn: int | None = None,
        staticnoise: int | None = None,
        cpuopt: int | None = None,
        patsize: int | None = None,
        tpdfo: int | None = None,
        tpdfn: int | None = None,
        corplane: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def histluma(self, clip: VideoNode, full: int | None = None, amp: int | None = None) -> ConstantFormatVideoNode: ...
    def matrix(
        self,
        clip: VideoNode,
        mat: _DataType | None = None,
        mats: _DataType | None = None,
        matd: _DataType | None = None,
        fulls: int | None = None,
        fulld: int | None = None,
        coef: _SingleAndSequence[float] | None = None,
        csp: int | None = None,
        col_fam: int | None = None,
        bits: int | None = None,
        singleout: int | None = None,
        cpuopt: int | None = None,
        planes: _SingleAndSequence[float] | None = None,
    ) -> ConstantFormatVideoNode: ...
    def matrix2020cl(
        self,
        clip: VideoNode,
        full: int | None = None,
        csp: int | None = None,
        bits: int | None = None,
        cpuopt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def nativetostack16(self, clip: VideoNode) -> ConstantFormatVideoNode: ...
    def primaries(
        self,
        clip: VideoNode,
        rs: _SingleAndSequence[float] | None = None,
        gs: _SingleAndSequence[float] | None = None,
        bs: _SingleAndSequence[float] | None = None,
        ws: _SingleAndSequence[float] | None = None,
        rd: _SingleAndSequence[float] | None = None,
        gd: _SingleAndSequence[float] | None = None,
        bd: _SingleAndSequence[float] | None = None,
        wd: _SingleAndSequence[float] | None = None,
        prims: _DataType | None = None,
        primd: _DataType | None = None,
        wconv: int | None = None,
        cpuopt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def resample(
        self,
        clip: VideoNode,
        w: int | None = None,
        h: int | None = None,
        sx: _SingleAndSequence[float] | None = None,
        sy: _SingleAndSequence[float] | None = None,
        sw: _SingleAndSequence[float] | None = None,
        sh: _SingleAndSequence[float] | None = None,
        scale: float | None = None,
        scaleh: float | None = None,
        scalev: float | None = None,
        kernel: _SingleAndSequence[_DataType] | None = None,
        kernelh: _SingleAndSequence[_DataType] | None = None,
        kernelv: _SingleAndSequence[_DataType] | None = None,
        impulse: _SingleAndSequence[float] | None = None,
        impulseh: _SingleAndSequence[float] | None = None,
        impulsev: _SingleAndSequence[float] | None = None,
        taps: _SingleAndSequence[int] | None = None,
        tapsh: _SingleAndSequence[int] | None = None,
        tapsv: _SingleAndSequence[int] | None = None,
        a1: _SingleAndSequence[float] | None = None,
        a2: _SingleAndSequence[float] | None = None,
        a3: _SingleAndSequence[float] | None = None,
        a1h: _SingleAndSequence[float] | None = None,
        a2h: _SingleAndSequence[float] | None = None,
        a3h: _SingleAndSequence[float] | None = None,
        a1v: _SingleAndSequence[float] | None = None,
        a2v: _SingleAndSequence[float] | None = None,
        a3v: _SingleAndSequence[float] | None = None,
        kovrspl: _SingleAndSequence[int] | None = None,
        fh: _SingleAndSequence[float] | None = None,
        fv: _SingleAndSequence[float] | None = None,
        cnorm: _SingleAndSequence[int] | None = None,
        total: _SingleAndSequence[float] | None = None,
        totalh: _SingleAndSequence[float] | None = None,
        totalv: _SingleAndSequence[float] | None = None,
        invks: _SingleAndSequence[int] | None = None,
        invksh: _SingleAndSequence[int] | None = None,
        invksv: _SingleAndSequence[int] | None = None,
        invkstaps: _SingleAndSequence[int] | None = None,
        invkstapsh: _SingleAndSequence[int] | None = None,
        invkstapsv: _SingleAndSequence[int] | None = None,
        csp: int | None = None,
        css: _DataType | None = None,
        planes: _SingleAndSequence[float] | None = None,
        fulls: int | None = None,
        fulld: int | None = None,
        center: _SingleAndSequence[int] | None = None,
        cplace: _DataType | None = None,
        cplaces: _DataType | None = None,
        cplaced: _DataType | None = None,
        interlaced: int | None = None,
        interlacedd: int | None = None,
        tff: int | None = None,
        tffd: int | None = None,
        flt: int | None = None,
        cpuopt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def stack16tonative(self, clip: VideoNode) -> ConstantFormatVideoNode: ...
    def transfer(
        self,
        clip: VideoNode,
        transs: _SingleAndSequence[_DataType] | None = None,
        transd: _SingleAndSequence[_DataType] | None = None,
        cont: float | None = None,
        gcor: float | None = None,
        bits: int | None = None,
        flt: int | None = None,
        fulls: int | None = None,
        fulld: int | None = None,
        logceis: int | None = None,
        logceid: int | None = None,
        cpuopt: int | None = None,
        blacklvl: float | None = None,
        sceneref: int | None = None,
        lb: float | None = None,
        lw: float | None = None,
        lws: float | None = None,
        lwd: float | None = None,
        ambient: float | None = None,
        match: int | None = None,
        gy: int | None = None,
        debug: int | None = None,
        sig_c: float | None = None,
        sig_t: float | None = None,
    ) -> ConstantFormatVideoNode: ...

class _Plugin_fmtc_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "fmtc" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def bitdepth(
        self,
        csp: int | None = None,
        bits: int | None = None,
        flt: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
        fulls: int | None = None,
        fulld: int | None = None,
        dmode: int | None = None,
        ampo: float | None = None,
        ampn: float | None = None,
        dyn: int | None = None,
        staticnoise: int | None = None,
        cpuopt: int | None = None,
        patsize: int | None = None,
        tpdfo: int | None = None,
        tpdfn: int | None = None,
        corplane: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def histluma(self, full: int | None = None, amp: int | None = None) -> ConstantFormatVideoNode: ...
    def matrix(
        self,
        mat: _DataType | None = None,
        mats: _DataType | None = None,
        matd: _DataType | None = None,
        fulls: int | None = None,
        fulld: int | None = None,
        coef: _SingleAndSequence[float] | None = None,
        csp: int | None = None,
        col_fam: int | None = None,
        bits: int | None = None,
        singleout: int | None = None,
        cpuopt: int | None = None,
        planes: _SingleAndSequence[float] | None = None,
    ) -> ConstantFormatVideoNode: ...
    def matrix2020cl(
        self, full: int | None = None, csp: int | None = None, bits: int | None = None, cpuopt: int | None = None
    ) -> ConstantFormatVideoNode: ...
    def nativetostack16(self) -> ConstantFormatVideoNode: ...
    def primaries(
        self,
        rs: _SingleAndSequence[float] | None = None,
        gs: _SingleAndSequence[float] | None = None,
        bs: _SingleAndSequence[float] | None = None,
        ws: _SingleAndSequence[float] | None = None,
        rd: _SingleAndSequence[float] | None = None,
        gd: _SingleAndSequence[float] | None = None,
        bd: _SingleAndSequence[float] | None = None,
        wd: _SingleAndSequence[float] | None = None,
        prims: _DataType | None = None,
        primd: _DataType | None = None,
        wconv: int | None = None,
        cpuopt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def resample(
        self,
        w: int | None = None,
        h: int | None = None,
        sx: _SingleAndSequence[float] | None = None,
        sy: _SingleAndSequence[float] | None = None,
        sw: _SingleAndSequence[float] | None = None,
        sh: _SingleAndSequence[float] | None = None,
        scale: float | None = None,
        scaleh: float | None = None,
        scalev: float | None = None,
        kernel: _SingleAndSequence[_DataType] | None = None,
        kernelh: _SingleAndSequence[_DataType] | None = None,
        kernelv: _SingleAndSequence[_DataType] | None = None,
        impulse: _SingleAndSequence[float] | None = None,
        impulseh: _SingleAndSequence[float] | None = None,
        impulsev: _SingleAndSequence[float] | None = None,
        taps: _SingleAndSequence[int] | None = None,
        tapsh: _SingleAndSequence[int] | None = None,
        tapsv: _SingleAndSequence[int] | None = None,
        a1: _SingleAndSequence[float] | None = None,
        a2: _SingleAndSequence[float] | None = None,
        a3: _SingleAndSequence[float] | None = None,
        a1h: _SingleAndSequence[float] | None = None,
        a2h: _SingleAndSequence[float] | None = None,
        a3h: _SingleAndSequence[float] | None = None,
        a1v: _SingleAndSequence[float] | None = None,
        a2v: _SingleAndSequence[float] | None = None,
        a3v: _SingleAndSequence[float] | None = None,
        kovrspl: _SingleAndSequence[int] | None = None,
        fh: _SingleAndSequence[float] | None = None,
        fv: _SingleAndSequence[float] | None = None,
        cnorm: _SingleAndSequence[int] | None = None,
        total: _SingleAndSequence[float] | None = None,
        totalh: _SingleAndSequence[float] | None = None,
        totalv: _SingleAndSequence[float] | None = None,
        invks: _SingleAndSequence[int] | None = None,
        invksh: _SingleAndSequence[int] | None = None,
        invksv: _SingleAndSequence[int] | None = None,
        invkstaps: _SingleAndSequence[int] | None = None,
        invkstapsh: _SingleAndSequence[int] | None = None,
        invkstapsv: _SingleAndSequence[int] | None = None,
        csp: int | None = None,
        css: _DataType | None = None,
        planes: _SingleAndSequence[float] | None = None,
        fulls: int | None = None,
        fulld: int | None = None,
        center: _SingleAndSequence[int] | None = None,
        cplace: _DataType | None = None,
        cplaces: _DataType | None = None,
        cplaced: _DataType | None = None,
        interlaced: int | None = None,
        interlacedd: int | None = None,
        tff: int | None = None,
        tffd: int | None = None,
        flt: int | None = None,
        cpuopt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def stack16tonative(self) -> ConstantFormatVideoNode: ...
    def transfer(
        self,
        transs: _SingleAndSequence[_DataType] | None = None,
        transd: _SingleAndSequence[_DataType] | None = None,
        cont: float | None = None,
        gcor: float | None = None,
        bits: int | None = None,
        flt: int | None = None,
        fulls: int | None = None,
        fulld: int | None = None,
        logceis: int | None = None,
        logceid: int | None = None,
        cpuopt: int | None = None,
        blacklvl: float | None = None,
        sceneref: int | None = None,
        lb: float | None = None,
        lw: float | None = None,
        lws: float | None = None,
        lwd: float | None = None,
        ambient: float | None = None,
        match: int | None = None,
        gy: int | None = None,
        debug: int | None = None,
        sig_c: float | None = None,
        sig_t: float | None = None,
    ) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: hysteresis

class _Plugin_hysteresis_Core_Bound(Plugin):
    """This class implements the module definitions for the "hysteresis" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Hysteresis(
        self, clipa: VideoNode, clipb: VideoNode, planes: _SingleAndSequence[int] | None = None
    ) -> ConstantFormatVideoNode: ...

class _Plugin_hysteresis_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "hysteresis" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Hysteresis(
        self, clipb: VideoNode, planes: _SingleAndSequence[int] | None = None
    ) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: imwri

class _Plugin_imwri_Core_Bound(Plugin):
    """This class implements the module definitions for the "imwri" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Read(
        self,
        filename: _SingleAndSequence[_DataType],
        firstnum: int | None = None,
        mismatch: int | None = None,
        alpha: int | None = None,
        float_output: int | None = None,
        embed_icc: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Write(
        self,
        clip: VideoNode,
        imgformat: _DataType,
        filename: _DataType,
        firstnum: int | None = None,
        quality: int | None = None,
        dither: int | None = None,
        compression_type: _DataType | None = None,
        overwrite: int | None = None,
        alpha: VideoNode | None = None,
    ) -> ConstantFormatVideoNode: ...

class _Plugin_imwri_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "imwri" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Write(
        self,
        imgformat: _DataType,
        filename: _DataType,
        firstnum: int | None = None,
        quality: int | None = None,
        dither: int | None = None,
        compression_type: _DataType | None = None,
        overwrite: int | None = None,
        alpha: VideoNode | None = None,
    ) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: knlm

class _Plugin_knlm_Core_Bound(Plugin):
    """This class implements the module definitions for the "knlm" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def KNLMeansCL(
        self,
        clip: VideoNode,
        d: int | None = None,
        a: int | None = None,
        s: int | None = None,
        h: float | None = None,
        channels: _DataType | None = None,
        wmode: int | None = None,
        wref: float | None = None,
        rclip: VideoNode | None = None,
        device_type: _DataType | None = None,
        device_id: int | None = None,
        ocl_x: int | None = None,
        ocl_y: int | None = None,
        ocl_r: int | None = None,
        info: int | None = None,
    ) -> ConstantFormatVideoNode: ...

class _Plugin_knlm_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "knlm" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def KNLMeansCL(
        self,
        d: int | None = None,
        a: int | None = None,
        s: int | None = None,
        h: float | None = None,
        channels: _DataType | None = None,
        wmode: int | None = None,
        wref: float | None = None,
        rclip: VideoNode | None = None,
        device_type: _DataType | None = None,
        device_id: int | None = None,
        ocl_x: int | None = None,
        ocl_y: int | None = None,
        ocl_r: int | None = None,
        info: int | None = None,
    ) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: lsmas

class _Plugin_lsmas_Core_Bound(Plugin):
    """This class implements the module definitions for the "lsmas" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def LibavSMASHSource(
        self,
        source: _DataType,
        track: int | None = None,
        threads: int | None = None,
        seek_mode: int | None = None,
        seek_threshold: int | None = None,
        dr: int | None = None,
        fpsnum: int | None = None,
        fpsden: int | None = None,
        variable: int | None = None,
        format: _DataType | None = None,
        decoder: _DataType | None = None,
        prefer_hw: int | None = None,
        ff_loglevel: int | None = None,
        ff_options: _DataType | None = None,
    ) -> VideoNode: ...
    def LWLibavSource(
        self,
        source: _DataType,
        stream_index: int | None = None,
        cache: int | None = None,
        cachefile: _DataType | None = None,
        threads: int | None = None,
        seek_mode: int | None = None,
        seek_threshold: int | None = None,
        dr: int | None = None,
        fpsnum: int | None = None,
        fpsden: int | None = None,
        variable: int | None = None,
        format: _DataType | None = None,
        decoder: _DataType | None = None,
        prefer_hw: int | None = None,
        repeat: int | None = None,
        dominance: int | None = None,
        ff_loglevel: int | None = None,
        cachedir: _DataType | None = None,
        ff_options: _DataType | None = None,
    ) -> VideoNode: ...

# end implementation

# implementation: manipmv

class _Plugin_manipmv_Core_Bound(Plugin):
    """This class implements the module definitions for the "manipmv" VapourSynth plugin.

    *This class cannot be imported.*"""
    def ExpandAnalysisData(self, clip: VideoNode) -> ConstantFormatVideoNode: ...
    def ScaleVect(
        self, clip: VideoNode, scaleX: int | None = None, scaleY: int | None = None
    ) -> ConstantFormatVideoNode: ...
    def ShowVect(
        self, clip: VideoNode, vectors: VideoNode, useSceneChangeProps: int | None = None
    ) -> ConstantFormatVideoNode: ...

class _Plugin_manipmv_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "manipmv" VapourSynth plugin.

    *This class cannot be imported.*"""
    def ExpandAnalysisData(self) -> ConstantFormatVideoNode: ...
    def ScaleVect(self, scaleX: int | None = None, scaleY: int | None = None) -> ConstantFormatVideoNode: ...
    def ShowVect(self, vectors: VideoNode, useSceneChangeProps: int | None = None) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: mv

class _Plugin_mv_Core_Bound(Plugin):
    """This class implements the module definitions for the "mv" VapourSynth plugin.

    *This class cannot be imported.*"""
    def Analyse(self, *args: _VapourSynthMapValue, **kwargs: _VapourSynthMapValue) -> ConstantFormatVideoNode: ...
    def BlockFPS(
        self,
        clip: VideoNode,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        num: int | None = None,
        den: int | None = None,
        mode: int | None = None,
        ml: float | None = None,
        blend: int | None = None,
        thscd1: int | None = None,
        thscd2: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Compensate(
        self,
        clip: VideoNode,
        super: VideoNode,
        vectors: VideoNode,
        scbehavior: int | None = None,
        thsad: int | None = None,
        fields: int | None = None,
        time: float | None = None,
        thscd1: int | None = None,
        thscd2: int | None = None,
        opt: int | None = None,
        tff: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain1(
        self,
        clip: VideoNode,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        thsad: int | None = None,
        thsadc: int | None = None,
        plane: int | None = None,
        limit: int | None = None,
        limitc: int | None = None,
        thscd1: int | None = None,
        thscd2: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain2(
        self,
        clip: VideoNode,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        thsad: int | None = None,
        thsadc: int | None = None,
        plane: int | None = None,
        limit: int | None = None,
        limitc: int | None = None,
        thscd1: int | None = None,
        thscd2: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain3(
        self,
        clip: VideoNode,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        thsad: int | None = None,
        thsadc: int | None = None,
        plane: int | None = None,
        limit: int | None = None,
        limitc: int | None = None,
        thscd1: int | None = None,
        thscd2: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain4(
        self,
        clip: VideoNode,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        thsad: int | None = None,
        thsadc: int | None = None,
        plane: int | None = None,
        limit: int | None = None,
        limitc: int | None = None,
        thscd1: int | None = None,
        thscd2: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain5(
        self,
        clip: VideoNode,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        thsad: int | None = None,
        thsadc: int | None = None,
        plane: int | None = None,
        limit: int | None = None,
        limitc: int | None = None,
        thscd1: int | None = None,
        thscd2: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain6(
        self,
        clip: VideoNode,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        thsad: int | None = None,
        thsadc: int | None = None,
        plane: int | None = None,
        limit: int | None = None,
        limitc: int | None = None,
        thscd1: int | None = None,
        thscd2: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def DepanAnalyse(
        self,
        clip: VideoNode,
        vectors: VideoNode,
        mask: VideoNode | None = None,
        zoom: int | None = None,
        rot: int | None = None,
        pixaspect: float | None = None,
        error: float | None = None,
        info: int | None = None,
        wrong: float | None = None,
        zerow: float | None = None,
        thscd1: int | None = None,
        thscd2: int | None = None,
        fields: int | None = None,
        tff: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def DepanCompensate(
        self,
        clip: VideoNode,
        data: VideoNode,
        offset: float | None = None,
        subpixel: int | None = None,
        pixaspect: float | None = None,
        matchfields: int | None = None,
        mirror: int | None = None,
        blur: int | None = None,
        info: int | None = None,
        fields: int | None = None,
        tff: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def DepanEstimate(
        self,
        clip: VideoNode,
        trust: float | None = None,
        winx: int | None = None,
        winy: int | None = None,
        wleft: int | None = None,
        wtop: int | None = None,
        dxmax: int | None = None,
        dymax: int | None = None,
        zoommax: float | None = None,
        stab: float | None = None,
        pixaspect: float | None = None,
        info: int | None = None,
        show: int | None = None,
        fields: int | None = None,
        tff: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def DepanStabilise(
        self,
        clip: VideoNode,
        data: VideoNode,
        cutoff: float | None = None,
        damping: float | None = None,
        initzoom: float | None = None,
        addzoom: int | None = None,
        prev: int | None = None,
        next: int | None = None,
        mirror: int | None = None,
        blur: int | None = None,
        dxmax: float | None = None,
        dymax: float | None = None,
        zoommax: float | None = None,
        rotmax: float | None = None,
        subpixel: int | None = None,
        pixaspect: float | None = None,
        fitlast: int | None = None,
        tzoom: float | None = None,
        info: int | None = None,
        method: int | None = None,
        fields: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Finest(self, super: VideoNode, opt: int | None = None) -> ConstantFormatVideoNode: ...
    def Flow(
        self,
        clip: VideoNode,
        super: VideoNode,
        vectors: VideoNode,
        time: float | None = None,
        mode: int | None = None,
        fields: int | None = None,
        thscd1: int | None = None,
        thscd2: int | None = None,
        opt: int | None = None,
        tff: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def FlowBlur(
        self,
        clip: VideoNode,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        blur: float | None = None,
        prec: int | None = None,
        thscd1: int | None = None,
        thscd2: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def FlowFPS(
        self,
        clip: VideoNode,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        num: int | None = None,
        den: int | None = None,
        mask: int | None = None,
        ml: float | None = None,
        blend: int | None = None,
        thscd1: int | None = None,
        thscd2: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def FlowInter(
        self,
        clip: VideoNode,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        time: float | None = None,
        ml: float | None = None,
        blend: int | None = None,
        thscd1: int | None = None,
        thscd2: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Mask(
        self,
        clip: VideoNode,
        vectors: VideoNode,
        ml: float | None = None,
        gamma: float | None = None,
        kind: int | None = None,
        time: float | None = None,
        ysc: int | None = None,
        thscd1: int | None = None,
        thscd2: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Recalculate(self, *args: _VapourSynthMapValue, **kwargs: _VapourSynthMapValue) -> ConstantFormatVideoNode: ...
    def SCDetection(
        self, clip: VideoNode, vectors: VideoNode, thscd1: int | None = None, thscd2: int | None = None
    ) -> ConstantFormatVideoNode: ...
    def Super(
        self,
        clip: VideoNode,
        hpad: int | None = None,
        vpad: int | None = None,
        pel: int | None = None,
        levels: int | None = None,
        chroma: int | None = None,
        sharp: int | None = None,
        rfilter: int | None = None,
        pelclip: VideoNode | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...

class _Plugin_mv_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "mv" VapourSynth plugin.

    *This class cannot be imported.*"""
    def Analyse(self, *args: _VapourSynthMapValue, **kwargs: _VapourSynthMapValue) -> ConstantFormatVideoNode: ...
    def BlockFPS(
        self,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        num: int | None = None,
        den: int | None = None,
        mode: int | None = None,
        ml: float | None = None,
        blend: int | None = None,
        thscd1: int | None = None,
        thscd2: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Compensate(
        self,
        super: VideoNode,
        vectors: VideoNode,
        scbehavior: int | None = None,
        thsad: int | None = None,
        fields: int | None = None,
        time: float | None = None,
        thscd1: int | None = None,
        thscd2: int | None = None,
        opt: int | None = None,
        tff: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain1(
        self,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        thsad: int | None = None,
        thsadc: int | None = None,
        plane: int | None = None,
        limit: int | None = None,
        limitc: int | None = None,
        thscd1: int | None = None,
        thscd2: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain2(
        self,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        thsad: int | None = None,
        thsadc: int | None = None,
        plane: int | None = None,
        limit: int | None = None,
        limitc: int | None = None,
        thscd1: int | None = None,
        thscd2: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain3(
        self,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        thsad: int | None = None,
        thsadc: int | None = None,
        plane: int | None = None,
        limit: int | None = None,
        limitc: int | None = None,
        thscd1: int | None = None,
        thscd2: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain4(
        self,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        thsad: int | None = None,
        thsadc: int | None = None,
        plane: int | None = None,
        limit: int | None = None,
        limitc: int | None = None,
        thscd1: int | None = None,
        thscd2: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain5(
        self,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        thsad: int | None = None,
        thsadc: int | None = None,
        plane: int | None = None,
        limit: int | None = None,
        limitc: int | None = None,
        thscd1: int | None = None,
        thscd2: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain6(
        self,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        thsad: int | None = None,
        thsadc: int | None = None,
        plane: int | None = None,
        limit: int | None = None,
        limitc: int | None = None,
        thscd1: int | None = None,
        thscd2: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def DepanAnalyse(
        self,
        vectors: VideoNode,
        mask: VideoNode | None = None,
        zoom: int | None = None,
        rot: int | None = None,
        pixaspect: float | None = None,
        error: float | None = None,
        info: int | None = None,
        wrong: float | None = None,
        zerow: float | None = None,
        thscd1: int | None = None,
        thscd2: int | None = None,
        fields: int | None = None,
        tff: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def DepanCompensate(
        self,
        data: VideoNode,
        offset: float | None = None,
        subpixel: int | None = None,
        pixaspect: float | None = None,
        matchfields: int | None = None,
        mirror: int | None = None,
        blur: int | None = None,
        info: int | None = None,
        fields: int | None = None,
        tff: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def DepanEstimate(
        self,
        trust: float | None = None,
        winx: int | None = None,
        winy: int | None = None,
        wleft: int | None = None,
        wtop: int | None = None,
        dxmax: int | None = None,
        dymax: int | None = None,
        zoommax: float | None = None,
        stab: float | None = None,
        pixaspect: float | None = None,
        info: int | None = None,
        show: int | None = None,
        fields: int | None = None,
        tff: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def DepanStabilise(
        self,
        data: VideoNode,
        cutoff: float | None = None,
        damping: float | None = None,
        initzoom: float | None = None,
        addzoom: int | None = None,
        prev: int | None = None,
        next: int | None = None,
        mirror: int | None = None,
        blur: int | None = None,
        dxmax: float | None = None,
        dymax: float | None = None,
        zoommax: float | None = None,
        rotmax: float | None = None,
        subpixel: int | None = None,
        pixaspect: float | None = None,
        fitlast: int | None = None,
        tzoom: float | None = None,
        info: int | None = None,
        method: int | None = None,
        fields: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Finest(self, opt: int | None = None) -> ConstantFormatVideoNode: ...
    def Flow(
        self,
        super: VideoNode,
        vectors: VideoNode,
        time: float | None = None,
        mode: int | None = None,
        fields: int | None = None,
        thscd1: int | None = None,
        thscd2: int | None = None,
        opt: int | None = None,
        tff: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def FlowBlur(
        self,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        blur: float | None = None,
        prec: int | None = None,
        thscd1: int | None = None,
        thscd2: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def FlowFPS(
        self,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        num: int | None = None,
        den: int | None = None,
        mask: int | None = None,
        ml: float | None = None,
        blend: int | None = None,
        thscd1: int | None = None,
        thscd2: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def FlowInter(
        self,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        time: float | None = None,
        ml: float | None = None,
        blend: int | None = None,
        thscd1: int | None = None,
        thscd2: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Mask(
        self,
        vectors: VideoNode,
        ml: float | None = None,
        gamma: float | None = None,
        kind: int | None = None,
        time: float | None = None,
        ysc: int | None = None,
        thscd1: int | None = None,
        thscd2: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Recalculate(self, *args: _VapourSynthMapValue, **kwargs: _VapourSynthMapValue) -> ConstantFormatVideoNode: ...
    def SCDetection(
        self, vectors: VideoNode, thscd1: int | None = None, thscd2: int | None = None
    ) -> ConstantFormatVideoNode: ...
    def Super(
        self,
        hpad: int | None = None,
        vpad: int | None = None,
        pel: int | None = None,
        levels: int | None = None,
        chroma: int | None = None,
        sharp: int | None = None,
        rfilter: int | None = None,
        pelclip: VideoNode | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: mvsf

class _Plugin_mvsf_Core_Bound(Plugin):
    """This class implements the module definitions for the "mvsf" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Analyse(self, *args: _VapourSynthMapValue, **kwargs: _VapourSynthMapValue) -> ConstantFormatVideoNode: ...
    def Analyze(self, *args: _VapourSynthMapValue, **kwargs: _VapourSynthMapValue) -> ConstantFormatVideoNode: ...
    def BlockFPS(
        self,
        clip: VideoNode,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        num: int | None = None,
        den: int | None = None,
        mode: int | None = None,
        ml: float | None = None,
        blend: int | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Compensate(
        self,
        clip: VideoNode,
        super: VideoNode,
        vectors: VideoNode,
        scbehavior: int | None = None,
        thsad: float | None = None,
        fields: int | None = None,
        time: float | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
        tff: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain1(
        self,
        clip: VideoNode,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain10(
        self,
        clip: VideoNode,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        mvbw7: VideoNode,
        mvfw7: VideoNode,
        mvbw8: VideoNode,
        mvfw8: VideoNode,
        mvbw9: VideoNode,
        mvfw9: VideoNode,
        mvbw10: VideoNode,
        mvfw10: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain11(
        self,
        clip: VideoNode,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        mvbw7: VideoNode,
        mvfw7: VideoNode,
        mvbw8: VideoNode,
        mvfw8: VideoNode,
        mvbw9: VideoNode,
        mvfw9: VideoNode,
        mvbw10: VideoNode,
        mvfw10: VideoNode,
        mvbw11: VideoNode,
        mvfw11: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain12(
        self,
        clip: VideoNode,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        mvbw7: VideoNode,
        mvfw7: VideoNode,
        mvbw8: VideoNode,
        mvfw8: VideoNode,
        mvbw9: VideoNode,
        mvfw9: VideoNode,
        mvbw10: VideoNode,
        mvfw10: VideoNode,
        mvbw11: VideoNode,
        mvfw11: VideoNode,
        mvbw12: VideoNode,
        mvfw12: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain13(
        self,
        clip: VideoNode,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        mvbw7: VideoNode,
        mvfw7: VideoNode,
        mvbw8: VideoNode,
        mvfw8: VideoNode,
        mvbw9: VideoNode,
        mvfw9: VideoNode,
        mvbw10: VideoNode,
        mvfw10: VideoNode,
        mvbw11: VideoNode,
        mvfw11: VideoNode,
        mvbw12: VideoNode,
        mvfw12: VideoNode,
        mvbw13: VideoNode,
        mvfw13: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain14(
        self,
        clip: VideoNode,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        mvbw7: VideoNode,
        mvfw7: VideoNode,
        mvbw8: VideoNode,
        mvfw8: VideoNode,
        mvbw9: VideoNode,
        mvfw9: VideoNode,
        mvbw10: VideoNode,
        mvfw10: VideoNode,
        mvbw11: VideoNode,
        mvfw11: VideoNode,
        mvbw12: VideoNode,
        mvfw12: VideoNode,
        mvbw13: VideoNode,
        mvfw13: VideoNode,
        mvbw14: VideoNode,
        mvfw14: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain15(
        self,
        clip: VideoNode,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        mvbw7: VideoNode,
        mvfw7: VideoNode,
        mvbw8: VideoNode,
        mvfw8: VideoNode,
        mvbw9: VideoNode,
        mvfw9: VideoNode,
        mvbw10: VideoNode,
        mvfw10: VideoNode,
        mvbw11: VideoNode,
        mvfw11: VideoNode,
        mvbw12: VideoNode,
        mvfw12: VideoNode,
        mvbw13: VideoNode,
        mvfw13: VideoNode,
        mvbw14: VideoNode,
        mvfw14: VideoNode,
        mvbw15: VideoNode,
        mvfw15: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain16(
        self,
        clip: VideoNode,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        mvbw7: VideoNode,
        mvfw7: VideoNode,
        mvbw8: VideoNode,
        mvfw8: VideoNode,
        mvbw9: VideoNode,
        mvfw9: VideoNode,
        mvbw10: VideoNode,
        mvfw10: VideoNode,
        mvbw11: VideoNode,
        mvfw11: VideoNode,
        mvbw12: VideoNode,
        mvfw12: VideoNode,
        mvbw13: VideoNode,
        mvfw13: VideoNode,
        mvbw14: VideoNode,
        mvfw14: VideoNode,
        mvbw15: VideoNode,
        mvfw15: VideoNode,
        mvbw16: VideoNode,
        mvfw16: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain17(
        self,
        clip: VideoNode,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        mvbw7: VideoNode,
        mvfw7: VideoNode,
        mvbw8: VideoNode,
        mvfw8: VideoNode,
        mvbw9: VideoNode,
        mvfw9: VideoNode,
        mvbw10: VideoNode,
        mvfw10: VideoNode,
        mvbw11: VideoNode,
        mvfw11: VideoNode,
        mvbw12: VideoNode,
        mvfw12: VideoNode,
        mvbw13: VideoNode,
        mvfw13: VideoNode,
        mvbw14: VideoNode,
        mvfw14: VideoNode,
        mvbw15: VideoNode,
        mvfw15: VideoNode,
        mvbw16: VideoNode,
        mvfw16: VideoNode,
        mvbw17: VideoNode,
        mvfw17: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain18(
        self,
        clip: VideoNode,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        mvbw7: VideoNode,
        mvfw7: VideoNode,
        mvbw8: VideoNode,
        mvfw8: VideoNode,
        mvbw9: VideoNode,
        mvfw9: VideoNode,
        mvbw10: VideoNode,
        mvfw10: VideoNode,
        mvbw11: VideoNode,
        mvfw11: VideoNode,
        mvbw12: VideoNode,
        mvfw12: VideoNode,
        mvbw13: VideoNode,
        mvfw13: VideoNode,
        mvbw14: VideoNode,
        mvfw14: VideoNode,
        mvbw15: VideoNode,
        mvfw15: VideoNode,
        mvbw16: VideoNode,
        mvfw16: VideoNode,
        mvbw17: VideoNode,
        mvfw17: VideoNode,
        mvbw18: VideoNode,
        mvfw18: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain19(
        self,
        clip: VideoNode,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        mvbw7: VideoNode,
        mvfw7: VideoNode,
        mvbw8: VideoNode,
        mvfw8: VideoNode,
        mvbw9: VideoNode,
        mvfw9: VideoNode,
        mvbw10: VideoNode,
        mvfw10: VideoNode,
        mvbw11: VideoNode,
        mvfw11: VideoNode,
        mvbw12: VideoNode,
        mvfw12: VideoNode,
        mvbw13: VideoNode,
        mvfw13: VideoNode,
        mvbw14: VideoNode,
        mvfw14: VideoNode,
        mvbw15: VideoNode,
        mvfw15: VideoNode,
        mvbw16: VideoNode,
        mvfw16: VideoNode,
        mvbw17: VideoNode,
        mvfw17: VideoNode,
        mvbw18: VideoNode,
        mvfw18: VideoNode,
        mvbw19: VideoNode,
        mvfw19: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain2(
        self,
        clip: VideoNode,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain20(
        self,
        clip: VideoNode,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        mvbw7: VideoNode,
        mvfw7: VideoNode,
        mvbw8: VideoNode,
        mvfw8: VideoNode,
        mvbw9: VideoNode,
        mvfw9: VideoNode,
        mvbw10: VideoNode,
        mvfw10: VideoNode,
        mvbw11: VideoNode,
        mvfw11: VideoNode,
        mvbw12: VideoNode,
        mvfw12: VideoNode,
        mvbw13: VideoNode,
        mvfw13: VideoNode,
        mvbw14: VideoNode,
        mvfw14: VideoNode,
        mvbw15: VideoNode,
        mvfw15: VideoNode,
        mvbw16: VideoNode,
        mvfw16: VideoNode,
        mvbw17: VideoNode,
        mvfw17: VideoNode,
        mvbw18: VideoNode,
        mvfw18: VideoNode,
        mvbw19: VideoNode,
        mvfw19: VideoNode,
        mvbw20: VideoNode,
        mvfw20: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain21(
        self,
        clip: VideoNode,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        mvbw7: VideoNode,
        mvfw7: VideoNode,
        mvbw8: VideoNode,
        mvfw8: VideoNode,
        mvbw9: VideoNode,
        mvfw9: VideoNode,
        mvbw10: VideoNode,
        mvfw10: VideoNode,
        mvbw11: VideoNode,
        mvfw11: VideoNode,
        mvbw12: VideoNode,
        mvfw12: VideoNode,
        mvbw13: VideoNode,
        mvfw13: VideoNode,
        mvbw14: VideoNode,
        mvfw14: VideoNode,
        mvbw15: VideoNode,
        mvfw15: VideoNode,
        mvbw16: VideoNode,
        mvfw16: VideoNode,
        mvbw17: VideoNode,
        mvfw17: VideoNode,
        mvbw18: VideoNode,
        mvfw18: VideoNode,
        mvbw19: VideoNode,
        mvfw19: VideoNode,
        mvbw20: VideoNode,
        mvfw20: VideoNode,
        mvbw21: VideoNode,
        mvfw21: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain22(
        self,
        clip: VideoNode,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        mvbw7: VideoNode,
        mvfw7: VideoNode,
        mvbw8: VideoNode,
        mvfw8: VideoNode,
        mvbw9: VideoNode,
        mvfw9: VideoNode,
        mvbw10: VideoNode,
        mvfw10: VideoNode,
        mvbw11: VideoNode,
        mvfw11: VideoNode,
        mvbw12: VideoNode,
        mvfw12: VideoNode,
        mvbw13: VideoNode,
        mvfw13: VideoNode,
        mvbw14: VideoNode,
        mvfw14: VideoNode,
        mvbw15: VideoNode,
        mvfw15: VideoNode,
        mvbw16: VideoNode,
        mvfw16: VideoNode,
        mvbw17: VideoNode,
        mvfw17: VideoNode,
        mvbw18: VideoNode,
        mvfw18: VideoNode,
        mvbw19: VideoNode,
        mvfw19: VideoNode,
        mvbw20: VideoNode,
        mvfw20: VideoNode,
        mvbw21: VideoNode,
        mvfw21: VideoNode,
        mvbw22: VideoNode,
        mvfw22: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain23(
        self,
        clip: VideoNode,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        mvbw7: VideoNode,
        mvfw7: VideoNode,
        mvbw8: VideoNode,
        mvfw8: VideoNode,
        mvbw9: VideoNode,
        mvfw9: VideoNode,
        mvbw10: VideoNode,
        mvfw10: VideoNode,
        mvbw11: VideoNode,
        mvfw11: VideoNode,
        mvbw12: VideoNode,
        mvfw12: VideoNode,
        mvbw13: VideoNode,
        mvfw13: VideoNode,
        mvbw14: VideoNode,
        mvfw14: VideoNode,
        mvbw15: VideoNode,
        mvfw15: VideoNode,
        mvbw16: VideoNode,
        mvfw16: VideoNode,
        mvbw17: VideoNode,
        mvfw17: VideoNode,
        mvbw18: VideoNode,
        mvfw18: VideoNode,
        mvbw19: VideoNode,
        mvfw19: VideoNode,
        mvbw20: VideoNode,
        mvfw20: VideoNode,
        mvbw21: VideoNode,
        mvfw21: VideoNode,
        mvbw22: VideoNode,
        mvfw22: VideoNode,
        mvbw23: VideoNode,
        mvfw23: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain24(
        self,
        clip: VideoNode,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        mvbw7: VideoNode,
        mvfw7: VideoNode,
        mvbw8: VideoNode,
        mvfw8: VideoNode,
        mvbw9: VideoNode,
        mvfw9: VideoNode,
        mvbw10: VideoNode,
        mvfw10: VideoNode,
        mvbw11: VideoNode,
        mvfw11: VideoNode,
        mvbw12: VideoNode,
        mvfw12: VideoNode,
        mvbw13: VideoNode,
        mvfw13: VideoNode,
        mvbw14: VideoNode,
        mvfw14: VideoNode,
        mvbw15: VideoNode,
        mvfw15: VideoNode,
        mvbw16: VideoNode,
        mvfw16: VideoNode,
        mvbw17: VideoNode,
        mvfw17: VideoNode,
        mvbw18: VideoNode,
        mvfw18: VideoNode,
        mvbw19: VideoNode,
        mvfw19: VideoNode,
        mvbw20: VideoNode,
        mvfw20: VideoNode,
        mvbw21: VideoNode,
        mvfw21: VideoNode,
        mvbw22: VideoNode,
        mvfw22: VideoNode,
        mvbw23: VideoNode,
        mvfw23: VideoNode,
        mvbw24: VideoNode,
        mvfw24: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain3(
        self,
        clip: VideoNode,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain4(
        self,
        clip: VideoNode,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain5(
        self,
        clip: VideoNode,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain6(
        self,
        clip: VideoNode,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain7(
        self,
        clip: VideoNode,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        mvbw7: VideoNode,
        mvfw7: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain8(
        self,
        clip: VideoNode,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        mvbw7: VideoNode,
        mvfw7: VideoNode,
        mvbw8: VideoNode,
        mvfw8: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain9(
        self,
        clip: VideoNode,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        mvbw7: VideoNode,
        mvfw7: VideoNode,
        mvbw8: VideoNode,
        mvfw8: VideoNode,
        mvbw9: VideoNode,
        mvfw9: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Finest(self, super: VideoNode) -> ConstantFormatVideoNode: ...
    def Flow(
        self,
        clip: VideoNode,
        super: VideoNode,
        vectors: VideoNode,
        time: float | None = None,
        mode: int | None = None,
        fields: int | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
        tff: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def FlowBlur(
        self,
        clip: VideoNode,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        blur: float | None = None,
        prec: int | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def FlowFPS(
        self,
        clip: VideoNode,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        num: int | None = None,
        den: int | None = None,
        mask: int | None = None,
        ml: float | None = None,
        blend: int | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def FlowInter(
        self,
        clip: VideoNode,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        time: float | None = None,
        ml: float | None = None,
        blend: int | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Mask(
        self,
        clip: VideoNode,
        vectors: VideoNode,
        ml: float | None = None,
        gamma: float | None = None,
        kind: int | None = None,
        time: float | None = None,
        ysc: float | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Recalculate(self, *args: _VapourSynthMapValue, **kwargs: _VapourSynthMapValue) -> ConstantFormatVideoNode: ...
    def SCDetection(
        self, clip: VideoNode, vectors: VideoNode, thscd1: float | None = None, thscd2: float | None = None
    ) -> ConstantFormatVideoNode: ...
    def Super(
        self,
        clip: VideoNode,
        hpad: int | None = None,
        vpad: int | None = None,
        pel: int | None = None,
        levels: int | None = None,
        chroma: int | None = None,
        sharp: int | None = None,
        rfilter: int | None = None,
        pelclip: VideoNode | None = None,
    ) -> ConstantFormatVideoNode: ...

class _Plugin_mvsf_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "mvsf" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Analyse(self, *args: _VapourSynthMapValue, **kwargs: _VapourSynthMapValue) -> ConstantFormatVideoNode: ...
    def Analyze(self, *args: _VapourSynthMapValue, **kwargs: _VapourSynthMapValue) -> ConstantFormatVideoNode: ...
    def BlockFPS(
        self,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        num: int | None = None,
        den: int | None = None,
        mode: int | None = None,
        ml: float | None = None,
        blend: int | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Compensate(
        self,
        super: VideoNode,
        vectors: VideoNode,
        scbehavior: int | None = None,
        thsad: float | None = None,
        fields: int | None = None,
        time: float | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
        tff: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain1(
        self,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain10(
        self,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        mvbw7: VideoNode,
        mvfw7: VideoNode,
        mvbw8: VideoNode,
        mvfw8: VideoNode,
        mvbw9: VideoNode,
        mvfw9: VideoNode,
        mvbw10: VideoNode,
        mvfw10: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain11(
        self,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        mvbw7: VideoNode,
        mvfw7: VideoNode,
        mvbw8: VideoNode,
        mvfw8: VideoNode,
        mvbw9: VideoNode,
        mvfw9: VideoNode,
        mvbw10: VideoNode,
        mvfw10: VideoNode,
        mvbw11: VideoNode,
        mvfw11: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain12(
        self,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        mvbw7: VideoNode,
        mvfw7: VideoNode,
        mvbw8: VideoNode,
        mvfw8: VideoNode,
        mvbw9: VideoNode,
        mvfw9: VideoNode,
        mvbw10: VideoNode,
        mvfw10: VideoNode,
        mvbw11: VideoNode,
        mvfw11: VideoNode,
        mvbw12: VideoNode,
        mvfw12: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain13(
        self,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        mvbw7: VideoNode,
        mvfw7: VideoNode,
        mvbw8: VideoNode,
        mvfw8: VideoNode,
        mvbw9: VideoNode,
        mvfw9: VideoNode,
        mvbw10: VideoNode,
        mvfw10: VideoNode,
        mvbw11: VideoNode,
        mvfw11: VideoNode,
        mvbw12: VideoNode,
        mvfw12: VideoNode,
        mvbw13: VideoNode,
        mvfw13: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain14(
        self,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        mvbw7: VideoNode,
        mvfw7: VideoNode,
        mvbw8: VideoNode,
        mvfw8: VideoNode,
        mvbw9: VideoNode,
        mvfw9: VideoNode,
        mvbw10: VideoNode,
        mvfw10: VideoNode,
        mvbw11: VideoNode,
        mvfw11: VideoNode,
        mvbw12: VideoNode,
        mvfw12: VideoNode,
        mvbw13: VideoNode,
        mvfw13: VideoNode,
        mvbw14: VideoNode,
        mvfw14: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain15(
        self,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        mvbw7: VideoNode,
        mvfw7: VideoNode,
        mvbw8: VideoNode,
        mvfw8: VideoNode,
        mvbw9: VideoNode,
        mvfw9: VideoNode,
        mvbw10: VideoNode,
        mvfw10: VideoNode,
        mvbw11: VideoNode,
        mvfw11: VideoNode,
        mvbw12: VideoNode,
        mvfw12: VideoNode,
        mvbw13: VideoNode,
        mvfw13: VideoNode,
        mvbw14: VideoNode,
        mvfw14: VideoNode,
        mvbw15: VideoNode,
        mvfw15: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain16(
        self,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        mvbw7: VideoNode,
        mvfw7: VideoNode,
        mvbw8: VideoNode,
        mvfw8: VideoNode,
        mvbw9: VideoNode,
        mvfw9: VideoNode,
        mvbw10: VideoNode,
        mvfw10: VideoNode,
        mvbw11: VideoNode,
        mvfw11: VideoNode,
        mvbw12: VideoNode,
        mvfw12: VideoNode,
        mvbw13: VideoNode,
        mvfw13: VideoNode,
        mvbw14: VideoNode,
        mvfw14: VideoNode,
        mvbw15: VideoNode,
        mvfw15: VideoNode,
        mvbw16: VideoNode,
        mvfw16: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain17(
        self,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        mvbw7: VideoNode,
        mvfw7: VideoNode,
        mvbw8: VideoNode,
        mvfw8: VideoNode,
        mvbw9: VideoNode,
        mvfw9: VideoNode,
        mvbw10: VideoNode,
        mvfw10: VideoNode,
        mvbw11: VideoNode,
        mvfw11: VideoNode,
        mvbw12: VideoNode,
        mvfw12: VideoNode,
        mvbw13: VideoNode,
        mvfw13: VideoNode,
        mvbw14: VideoNode,
        mvfw14: VideoNode,
        mvbw15: VideoNode,
        mvfw15: VideoNode,
        mvbw16: VideoNode,
        mvfw16: VideoNode,
        mvbw17: VideoNode,
        mvfw17: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain18(
        self,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        mvbw7: VideoNode,
        mvfw7: VideoNode,
        mvbw8: VideoNode,
        mvfw8: VideoNode,
        mvbw9: VideoNode,
        mvfw9: VideoNode,
        mvbw10: VideoNode,
        mvfw10: VideoNode,
        mvbw11: VideoNode,
        mvfw11: VideoNode,
        mvbw12: VideoNode,
        mvfw12: VideoNode,
        mvbw13: VideoNode,
        mvfw13: VideoNode,
        mvbw14: VideoNode,
        mvfw14: VideoNode,
        mvbw15: VideoNode,
        mvfw15: VideoNode,
        mvbw16: VideoNode,
        mvfw16: VideoNode,
        mvbw17: VideoNode,
        mvfw17: VideoNode,
        mvbw18: VideoNode,
        mvfw18: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain19(
        self,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        mvbw7: VideoNode,
        mvfw7: VideoNode,
        mvbw8: VideoNode,
        mvfw8: VideoNode,
        mvbw9: VideoNode,
        mvfw9: VideoNode,
        mvbw10: VideoNode,
        mvfw10: VideoNode,
        mvbw11: VideoNode,
        mvfw11: VideoNode,
        mvbw12: VideoNode,
        mvfw12: VideoNode,
        mvbw13: VideoNode,
        mvfw13: VideoNode,
        mvbw14: VideoNode,
        mvfw14: VideoNode,
        mvbw15: VideoNode,
        mvfw15: VideoNode,
        mvbw16: VideoNode,
        mvfw16: VideoNode,
        mvbw17: VideoNode,
        mvfw17: VideoNode,
        mvbw18: VideoNode,
        mvfw18: VideoNode,
        mvbw19: VideoNode,
        mvfw19: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain2(
        self,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain20(
        self,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        mvbw7: VideoNode,
        mvfw7: VideoNode,
        mvbw8: VideoNode,
        mvfw8: VideoNode,
        mvbw9: VideoNode,
        mvfw9: VideoNode,
        mvbw10: VideoNode,
        mvfw10: VideoNode,
        mvbw11: VideoNode,
        mvfw11: VideoNode,
        mvbw12: VideoNode,
        mvfw12: VideoNode,
        mvbw13: VideoNode,
        mvfw13: VideoNode,
        mvbw14: VideoNode,
        mvfw14: VideoNode,
        mvbw15: VideoNode,
        mvfw15: VideoNode,
        mvbw16: VideoNode,
        mvfw16: VideoNode,
        mvbw17: VideoNode,
        mvfw17: VideoNode,
        mvbw18: VideoNode,
        mvfw18: VideoNode,
        mvbw19: VideoNode,
        mvfw19: VideoNode,
        mvbw20: VideoNode,
        mvfw20: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain21(
        self,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        mvbw7: VideoNode,
        mvfw7: VideoNode,
        mvbw8: VideoNode,
        mvfw8: VideoNode,
        mvbw9: VideoNode,
        mvfw9: VideoNode,
        mvbw10: VideoNode,
        mvfw10: VideoNode,
        mvbw11: VideoNode,
        mvfw11: VideoNode,
        mvbw12: VideoNode,
        mvfw12: VideoNode,
        mvbw13: VideoNode,
        mvfw13: VideoNode,
        mvbw14: VideoNode,
        mvfw14: VideoNode,
        mvbw15: VideoNode,
        mvfw15: VideoNode,
        mvbw16: VideoNode,
        mvfw16: VideoNode,
        mvbw17: VideoNode,
        mvfw17: VideoNode,
        mvbw18: VideoNode,
        mvfw18: VideoNode,
        mvbw19: VideoNode,
        mvfw19: VideoNode,
        mvbw20: VideoNode,
        mvfw20: VideoNode,
        mvbw21: VideoNode,
        mvfw21: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain22(
        self,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        mvbw7: VideoNode,
        mvfw7: VideoNode,
        mvbw8: VideoNode,
        mvfw8: VideoNode,
        mvbw9: VideoNode,
        mvfw9: VideoNode,
        mvbw10: VideoNode,
        mvfw10: VideoNode,
        mvbw11: VideoNode,
        mvfw11: VideoNode,
        mvbw12: VideoNode,
        mvfw12: VideoNode,
        mvbw13: VideoNode,
        mvfw13: VideoNode,
        mvbw14: VideoNode,
        mvfw14: VideoNode,
        mvbw15: VideoNode,
        mvfw15: VideoNode,
        mvbw16: VideoNode,
        mvfw16: VideoNode,
        mvbw17: VideoNode,
        mvfw17: VideoNode,
        mvbw18: VideoNode,
        mvfw18: VideoNode,
        mvbw19: VideoNode,
        mvfw19: VideoNode,
        mvbw20: VideoNode,
        mvfw20: VideoNode,
        mvbw21: VideoNode,
        mvfw21: VideoNode,
        mvbw22: VideoNode,
        mvfw22: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain23(
        self,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        mvbw7: VideoNode,
        mvfw7: VideoNode,
        mvbw8: VideoNode,
        mvfw8: VideoNode,
        mvbw9: VideoNode,
        mvfw9: VideoNode,
        mvbw10: VideoNode,
        mvfw10: VideoNode,
        mvbw11: VideoNode,
        mvfw11: VideoNode,
        mvbw12: VideoNode,
        mvfw12: VideoNode,
        mvbw13: VideoNode,
        mvfw13: VideoNode,
        mvbw14: VideoNode,
        mvfw14: VideoNode,
        mvbw15: VideoNode,
        mvfw15: VideoNode,
        mvbw16: VideoNode,
        mvfw16: VideoNode,
        mvbw17: VideoNode,
        mvfw17: VideoNode,
        mvbw18: VideoNode,
        mvfw18: VideoNode,
        mvbw19: VideoNode,
        mvfw19: VideoNode,
        mvbw20: VideoNode,
        mvfw20: VideoNode,
        mvbw21: VideoNode,
        mvfw21: VideoNode,
        mvbw22: VideoNode,
        mvfw22: VideoNode,
        mvbw23: VideoNode,
        mvfw23: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain24(
        self,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        mvbw7: VideoNode,
        mvfw7: VideoNode,
        mvbw8: VideoNode,
        mvfw8: VideoNode,
        mvbw9: VideoNode,
        mvfw9: VideoNode,
        mvbw10: VideoNode,
        mvfw10: VideoNode,
        mvbw11: VideoNode,
        mvfw11: VideoNode,
        mvbw12: VideoNode,
        mvfw12: VideoNode,
        mvbw13: VideoNode,
        mvfw13: VideoNode,
        mvbw14: VideoNode,
        mvfw14: VideoNode,
        mvbw15: VideoNode,
        mvfw15: VideoNode,
        mvbw16: VideoNode,
        mvfw16: VideoNode,
        mvbw17: VideoNode,
        mvfw17: VideoNode,
        mvbw18: VideoNode,
        mvfw18: VideoNode,
        mvbw19: VideoNode,
        mvfw19: VideoNode,
        mvbw20: VideoNode,
        mvfw20: VideoNode,
        mvbw21: VideoNode,
        mvfw21: VideoNode,
        mvbw22: VideoNode,
        mvfw22: VideoNode,
        mvbw23: VideoNode,
        mvfw23: VideoNode,
        mvbw24: VideoNode,
        mvfw24: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain3(
        self,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain4(
        self,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain5(
        self,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain6(
        self,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain7(
        self,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        mvbw7: VideoNode,
        mvfw7: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain8(
        self,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        mvbw7: VideoNode,
        mvfw7: VideoNode,
        mvbw8: VideoNode,
        mvfw8: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Degrain9(
        self,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        mvbw2: VideoNode,
        mvfw2: VideoNode,
        mvbw3: VideoNode,
        mvfw3: VideoNode,
        mvbw4: VideoNode,
        mvfw4: VideoNode,
        mvbw5: VideoNode,
        mvfw5: VideoNode,
        mvbw6: VideoNode,
        mvfw6: VideoNode,
        mvbw7: VideoNode,
        mvfw7: VideoNode,
        mvbw8: VideoNode,
        mvfw8: VideoNode,
        mvbw9: VideoNode,
        mvfw9: VideoNode,
        thsad: _SingleAndSequence[float] | None = None,
        plane: int | None = None,
        limit: _SingleAndSequence[float] | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Finest(self) -> ConstantFormatVideoNode: ...
    def Flow(
        self,
        super: VideoNode,
        vectors: VideoNode,
        time: float | None = None,
        mode: int | None = None,
        fields: int | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
        tff: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def FlowBlur(
        self,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        blur: float | None = None,
        prec: int | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def FlowFPS(
        self,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        num: int | None = None,
        den: int | None = None,
        mask: int | None = None,
        ml: float | None = None,
        blend: int | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def FlowInter(
        self,
        super: VideoNode,
        mvbw: VideoNode,
        mvfw: VideoNode,
        time: float | None = None,
        ml: float | None = None,
        blend: int | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Mask(
        self,
        vectors: VideoNode,
        ml: float | None = None,
        gamma: float | None = None,
        kind: int | None = None,
        time: float | None = None,
        ysc: float | None = None,
        thscd1: float | None = None,
        thscd2: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Recalculate(self, *args: _VapourSynthMapValue, **kwargs: _VapourSynthMapValue) -> ConstantFormatVideoNode: ...
    def SCDetection(
        self, vectors: VideoNode, thscd1: float | None = None, thscd2: float | None = None
    ) -> ConstantFormatVideoNode: ...
    def Super(
        self,
        hpad: int | None = None,
        vpad: int | None = None,
        pel: int | None = None,
        levels: int | None = None,
        chroma: int | None = None,
        sharp: int | None = None,
        rfilter: int | None = None,
        pelclip: VideoNode | None = None,
    ) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: neo_f3kdb

class _Plugin_neo_f3kdb_Core_Bound(Plugin):
    """This class implements the module definitions for the "neo_f3kdb" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Deband(
        self,
        clip: VideoNode,
        range: int | None = None,
        y: int | None = None,
        cb: int | None = None,
        cr: int | None = None,
        grainy: int | None = None,
        grainc: int | None = None,
        sample_mode: int | None = None,
        seed: int | None = None,
        blur_first: int | None = None,
        dynamic_grain: int | None = None,
        opt: int | None = None,
        mt: int | None = None,
        dither_algo: int | None = None,
        keep_tv_range: int | None = None,
        output_depth: int | None = None,
        random_algo_ref: int | None = None,
        random_algo_grain: int | None = None,
        random_param_ref: float | None = None,
        random_param_grain: float | None = None,
        preset: _DataType | None = None,
        y_1: int | None = None,
        cb_1: int | None = None,
        cr_1: int | None = None,
        y_2: int | None = None,
        cb_2: int | None = None,
        cr_2: int | None = None,
        scale: int | None = None,
        angle_boost: float | None = None,
        max_angle: float | None = None,
    ) -> ConstantFormatVideoNode: ...

class _Plugin_neo_f3kdb_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "neo_f3kdb" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Deband(
        self,
        range: int | None = None,
        y: int | None = None,
        cb: int | None = None,
        cr: int | None = None,
        grainy: int | None = None,
        grainc: int | None = None,
        sample_mode: int | None = None,
        seed: int | None = None,
        blur_first: int | None = None,
        dynamic_grain: int | None = None,
        opt: int | None = None,
        mt: int | None = None,
        dither_algo: int | None = None,
        keep_tv_range: int | None = None,
        output_depth: int | None = None,
        random_algo_ref: int | None = None,
        random_algo_grain: int | None = None,
        random_param_ref: float | None = None,
        random_param_grain: float | None = None,
        preset: _DataType | None = None,
        y_1: int | None = None,
        cb_1: int | None = None,
        cr_1: int | None = None,
        y_2: int | None = None,
        cb_2: int | None = None,
        cr_2: int | None = None,
        scale: int | None = None,
        angle_boost: float | None = None,
        max_angle: float | None = None,
    ) -> ConstantFormatVideoNode: ...

# end implementation

_ReturnDict_nlm_cuda_Version = TypedDict(
    "_ReturnDict_nlm_cuda_Version",
    {
        "cuda_version": int,
        "version": bytes,
    },
)

# implementation: nlm_cuda

class _Plugin_nlm_cuda_Core_Bound(Plugin):
    """This class implements the module definitions for the "nlm_cuda" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def NLMeans(
        self,
        clip: VideoNode,
        d: int | None = None,
        a: int | None = None,
        s: int | None = None,
        h: float | None = None,
        channels: _DataType | None = None,
        wmode: int | None = None,
        wref: float | None = None,
        rclip: VideoNode | None = None,
        device_id: int | None = None,
        num_streams: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Version(self) -> _ReturnDict_nlm_cuda_Version: ...

class _Plugin_nlm_cuda_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "nlm_cuda" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def NLMeans(
        self,
        d: int | None = None,
        a: int | None = None,
        s: int | None = None,
        h: float | None = None,
        channels: _DataType | None = None,
        wmode: int | None = None,
        wref: float | None = None,
        rclip: VideoNode | None = None,
        device_id: int | None = None,
        num_streams: int | None = None,
    ) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: nlm_ispc

class _Plugin_nlm_ispc_Core_Bound(Plugin):
    """This class implements the module definitions for the "nlm_ispc" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def NLMeans(
        self,
        clip: VideoNode,
        d: int | None = None,
        a: int | None = None,
        s: int | None = None,
        h: float | None = None,
        channels: _DataType | None = None,
        wmode: int | None = None,
        wref: float | None = None,
        rclip: VideoNode | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Version(self) -> bytes: ...

class _Plugin_nlm_ispc_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "nlm_ispc" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def NLMeans(
        self,
        d: int | None = None,
        a: int | None = None,
        s: int | None = None,
        h: float | None = None,
        channels: _DataType | None = None,
        wmode: int | None = None,
        wref: float | None = None,
        rclip: VideoNode | None = None,
    ) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: noise

class _Plugin_noise_Core_Bound(Plugin):
    """This class implements the module definitions for the "noise" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Add(
        self,
        clip: VideoNode,
        var: float | None = None,
        uvar: float | None = None,
        type: int | None = None,
        hcorr: float | None = None,
        vcorr: float | None = None,
        xsize: float | None = None,
        ysize: float | None = None,
        scale: float | None = None,
        seed: int | None = None,
        constant: int | None = None,
        every: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...

class _Plugin_noise_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "noise" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Add(
        self,
        var: float | None = None,
        uvar: float | None = None,
        type: int | None = None,
        hcorr: float | None = None,
        vcorr: float | None = None,
        xsize: float | None = None,
        ysize: float | None = None,
        scale: float | None = None,
        seed: int | None = None,
        constant: int | None = None,
        every: int | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: placebo

class _Plugin_placebo_Core_Bound(Plugin):
    """This class implements the module definitions for the "placebo" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Deband(
        self,
        clip: VideoNode,
        planes: int | None = None,
        iterations: int | None = None,
        threshold: float | None = None,
        radius: float | None = None,
        grain: float | None = None,
        dither: int | None = None,
        dither_algo: int | None = None,
        log_level: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Resample(
        self,
        clip: VideoNode,
        width: int,
        height: int,
        filter: _DataType | None = None,
        clamp: float | None = None,
        blur: float | None = None,
        taper: float | None = None,
        radius: float | None = None,
        param1: float | None = None,
        param2: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        sx: float | None = None,
        sy: float | None = None,
        antiring: float | None = None,
        sigmoidize: int | None = None,
        sigmoid_center: float | None = None,
        sigmoid_slope: float | None = None,
        linearize: int | None = None,
        trc: int | None = None,
        min_luma: float | None = None,
        log_level: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Shader(
        self,
        clip: VideoNode,
        shader: _DataType | None = None,
        width: int | None = None,
        height: int | None = None,
        chroma_loc: int | None = None,
        matrix: int | None = None,
        trc: int | None = None,
        linearize: int | None = None,
        sigmoidize: int | None = None,
        sigmoid_center: float | None = None,
        sigmoid_slope: float | None = None,
        antiring: float | None = None,
        filter: _DataType | None = None,
        clamp: float | None = None,
        blur: float | None = None,
        taper: float | None = None,
        radius: float | None = None,
        param1: float | None = None,
        param2: float | None = None,
        shader_s: _DataType | None = None,
        log_level: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Tonemap(
        self,
        clip: VideoNode,
        src_csp: int | None = None,
        dst_csp: int | None = None,
        dst_prim: int | None = None,
        src_max: float | None = None,
        src_min: float | None = None,
        dst_max: float | None = None,
        dst_min: float | None = None,
        dynamic_peak_detection: int | None = None,
        smoothing_period: float | None = None,
        scene_threshold_low: float | None = None,
        scene_threshold_high: float | None = None,
        percentile: float | None = None,
        gamut_mapping: int | None = None,
        tone_mapping_function: int | None = None,
        tone_mapping_function_s: _DataType | None = None,
        tone_mapping_param: float | None = None,
        metadata: int | None = None,
        use_dovi: int | None = None,
        visualize_lut: int | None = None,
        show_clipping: int | None = None,
        contrast_recovery: float | None = None,
        log_level: int | None = None,
    ) -> ConstantFormatVideoNode: ...

class _Plugin_placebo_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "placebo" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Deband(
        self,
        planes: int | None = None,
        iterations: int | None = None,
        threshold: float | None = None,
        radius: float | None = None,
        grain: float | None = None,
        dither: int | None = None,
        dither_algo: int | None = None,
        log_level: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Resample(
        self,
        width: int,
        height: int,
        filter: _DataType | None = None,
        clamp: float | None = None,
        blur: float | None = None,
        taper: float | None = None,
        radius: float | None = None,
        param1: float | None = None,
        param2: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        sx: float | None = None,
        sy: float | None = None,
        antiring: float | None = None,
        sigmoidize: int | None = None,
        sigmoid_center: float | None = None,
        sigmoid_slope: float | None = None,
        linearize: int | None = None,
        trc: int | None = None,
        min_luma: float | None = None,
        log_level: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Shader(
        self,
        shader: _DataType | None = None,
        width: int | None = None,
        height: int | None = None,
        chroma_loc: int | None = None,
        matrix: int | None = None,
        trc: int | None = None,
        linearize: int | None = None,
        sigmoidize: int | None = None,
        sigmoid_center: float | None = None,
        sigmoid_slope: float | None = None,
        antiring: float | None = None,
        filter: _DataType | None = None,
        clamp: float | None = None,
        blur: float | None = None,
        taper: float | None = None,
        radius: float | None = None,
        param1: float | None = None,
        param2: float | None = None,
        shader_s: _DataType | None = None,
        log_level: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Tonemap(
        self,
        src_csp: int | None = None,
        dst_csp: int | None = None,
        dst_prim: int | None = None,
        src_max: float | None = None,
        src_min: float | None = None,
        dst_max: float | None = None,
        dst_min: float | None = None,
        dynamic_peak_detection: int | None = None,
        smoothing_period: float | None = None,
        scene_threshold_low: float | None = None,
        scene_threshold_high: float | None = None,
        percentile: float | None = None,
        gamut_mapping: int | None = None,
        tone_mapping_function: int | None = None,
        tone_mapping_function_s: _DataType | None = None,
        tone_mapping_param: float | None = None,
        metadata: int | None = None,
        use_dovi: int | None = None,
        visualize_lut: int | None = None,
        show_clipping: int | None = None,
        contrast_recovery: float | None = None,
        log_level: int | None = None,
    ) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: resize

class _Plugin_resize_Core_Bound(Plugin):
    """This class implements the module definitions for the "resize" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    @overload
    def Bicubic(
        self,
        clip: _VideoNodeT,
        width: int | None = None,
        height: int | None = None,
        format: None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> _VideoNodeT: ...
    @overload
    def Bicubic(
        self,
        clip: VideoNode,
        width: int | None = None,
        height: int | None = None,
        format: int = ...,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    @overload
    def Bicubic(
        self,
        clip: _VideoNodeT,
        width: int | None = None,
        height: int | None = None,
        format: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> _VideoNodeT: ...
    @overload
    def Bilinear(
        self,
        clip: _VideoNodeT,
        width: int | None = None,
        height: int | None = None,
        format: None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> _VideoNodeT: ...
    @overload
    def Bilinear(
        self,
        clip: VideoNode,
        width: int | None = None,
        height: int | None = None,
        format: int = ...,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    @overload
    def Bilinear(
        self,
        clip: _VideoNodeT,
        width: int | None = None,
        height: int | None = None,
        format: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> _VideoNodeT: ...
    def Bob(
        self,
        clip: VideoNode,
        filter: _DataType | None = None,
        tff: int | None = None,
        format: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    @overload
    def Lanczos(
        self,
        clip: _VideoNodeT,
        width: int | None = None,
        height: int | None = None,
        format: None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> _VideoNodeT: ...
    @overload
    def Lanczos(
        self,
        clip: VideoNode,
        width: int | None = None,
        height: int | None = None,
        format: int = ...,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    @overload
    def Lanczos(
        self,
        clip: _VideoNodeT,
        width: int | None = None,
        height: int | None = None,
        format: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> _VideoNodeT: ...
    @overload
    def Point(
        self,
        clip: _VideoNodeT,
        width: int | None = None,
        height: int | None = None,
        format: None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> _VideoNodeT: ...
    @overload
    def Point(
        self,
        clip: VideoNode,
        width: int | None = None,
        height: int | None = None,
        format: int = ...,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    @overload
    def Point(
        self,
        clip: _VideoNodeT,
        width: int | None = None,
        height: int | None = None,
        format: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> _VideoNodeT: ...
    @overload
    def Spline16(
        self,
        clip: _VideoNodeT,
        width: int | None = None,
        height: int | None = None,
        format: None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> _VideoNodeT: ...
    @overload
    def Spline16(
        self,
        clip: VideoNode,
        width: int | None = None,
        height: int | None = None,
        format: int = ...,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    @overload
    def Spline16(
        self,
        clip: _VideoNodeT,
        width: int | None = None,
        height: int | None = None,
        format: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> _VideoNodeT: ...
    @overload
    def Spline36(
        self,
        clip: _VideoNodeT,
        width: int | None = None,
        height: int | None = None,
        format: None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> _VideoNodeT: ...
    @overload
    def Spline36(
        self,
        clip: VideoNode,
        width: int | None = None,
        height: int | None = None,
        format: int = ...,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    @overload
    def Spline36(
        self,
        clip: _VideoNodeT,
        width: int | None = None,
        height: int | None = None,
        format: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> _VideoNodeT: ...
    @overload
    def Spline64(
        self,
        clip: _VideoNodeT,
        width: int | None = None,
        height: int | None = None,
        format: None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> _VideoNodeT: ...
    @overload
    def Spline64(
        self,
        clip: VideoNode,
        width: int | None = None,
        height: int | None = None,
        format: int = ...,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    @overload
    def Spline64(
        self,
        clip: _VideoNodeT,
        width: int | None = None,
        height: int | None = None,
        format: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> _VideoNodeT: ...

class _Plugin_resize_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "resize" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    @overload
    def Bicubic(
        self,
        width: int | None = None,
        height: int | None = None,
        format: None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> VideoNode: ...
    @overload
    def Bicubic(
        self,
        width: int | None = None,
        height: int | None = None,
        format: int = ...,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    @overload
    def Bicubic(
        self,
        width: int | None = None,
        height: int | None = None,
        format: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> VideoNode: ...
    @overload
    def Bilinear(
        self,
        width: int | None = None,
        height: int | None = None,
        format: None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> VideoNode: ...
    @overload
    def Bilinear(
        self,
        width: int | None = None,
        height: int | None = None,
        format: int = ...,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    @overload
    def Bilinear(
        self,
        width: int | None = None,
        height: int | None = None,
        format: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> VideoNode: ...
    def Bob(
        self,
        filter: _DataType | None = None,
        tff: int | None = None,
        format: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    @overload
    def Lanczos(
        self,
        width: int | None = None,
        height: int | None = None,
        format: None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> VideoNode: ...
    @overload
    def Lanczos(
        self,
        width: int | None = None,
        height: int | None = None,
        format: int = ...,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    @overload
    def Lanczos(
        self,
        width: int | None = None,
        height: int | None = None,
        format: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> VideoNode: ...
    @overload
    def Point(
        self,
        width: int | None = None,
        height: int | None = None,
        format: None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> VideoNode: ...
    @overload
    def Point(
        self,
        width: int | None = None,
        height: int | None = None,
        format: int = ...,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    @overload
    def Point(
        self,
        width: int | None = None,
        height: int | None = None,
        format: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> VideoNode: ...
    @overload
    def Spline16(
        self,
        width: int | None = None,
        height: int | None = None,
        format: None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> VideoNode: ...
    @overload
    def Spline16(
        self,
        width: int | None = None,
        height: int | None = None,
        format: int = ...,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    @overload
    def Spline16(
        self,
        width: int | None = None,
        height: int | None = None,
        format: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> VideoNode: ...
    @overload
    def Spline36(
        self,
        width: int | None = None,
        height: int | None = None,
        format: None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> VideoNode: ...
    @overload
    def Spline36(
        self,
        width: int | None = None,
        height: int | None = None,
        format: int = ...,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    @overload
    def Spline36(
        self,
        width: int | None = None,
        height: int | None = None,
        format: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> VideoNode: ...
    @overload
    def Spline64(
        self,
        width: int | None = None,
        height: int | None = None,
        format: None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> VideoNode: ...
    @overload
    def Spline64(
        self,
        width: int | None = None,
        height: int | None = None,
        format: int = ...,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    @overload
    def Spline64(
        self,
        width: int | None = None,
        height: int | None = None,
        format: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        approximate_gamma: int | None = None,
    ) -> VideoNode: ...

# end implementation

# implementation: resize2

class _CustomKernelCallback(Protocol):
    def __call__(self, *, x: float) -> float: ...

class _Plugin_resize2_Core_Bound(Plugin):
    """This class implements the module definitions for the "resize2" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    @overload
    def Bicubic(
        self,
        clip: _VideoNodeT,
        width: int | None = None,
        height: int | None = None,
        format: None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> _VideoNodeT: ...
    @overload
    def Bicubic(
        self,
        clip: VideoNode,
        width: int | None = None,
        height: int | None = None,
        format: int = ...,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    @overload
    def Bicubic(
        self,
        clip: _VideoNodeT,
        width: int | None = None,
        height: int | None = None,
        format: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> _VideoNodeT: ...
    @overload
    def Bilinear(
        self,
        clip: _VideoNodeT,
        width: int | None = None,
        height: int | None = None,
        format: None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> _VideoNodeT: ...
    @overload
    def Bilinear(
        self,
        clip: VideoNode,
        width: int | None = None,
        height: int | None = None,
        format: int = ...,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    @overload
    def Bilinear(
        self,
        clip: _VideoNodeT,
        width: int | None = None,
        height: int | None = None,
        format: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> _VideoNodeT: ...
    @overload
    def Lanczos(
        self,
        clip: _VideoNodeT,
        width: int | None = None,
        height: int | None = None,
        format: None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> _VideoNodeT: ...
    @overload
    def Lanczos(
        self,
        clip: VideoNode,
        width: int | None = None,
        height: int | None = None,
        format: int = ...,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    @overload
    def Lanczos(
        self,
        clip: _VideoNodeT,
        width: int | None = None,
        height: int | None = None,
        format: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> _VideoNodeT: ...
    @overload
    def Point(
        self,
        clip: _VideoNodeT,
        width: int | None = None,
        height: int | None = None,
        format: None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> _VideoNodeT: ...
    @overload
    def Point(
        self,
        clip: VideoNode,
        width: int | None = None,
        height: int | None = None,
        format: int = ...,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    @overload
    def Point(
        self,
        clip: _VideoNodeT,
        width: int | None = None,
        height: int | None = None,
        format: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> _VideoNodeT: ...
    @overload
    def Spline16(
        self,
        clip: _VideoNodeT,
        width: int | None = None,
        height: int | None = None,
        format: None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> _VideoNodeT: ...
    @overload
    def Spline16(
        self,
        clip: VideoNode,
        width: int | None = None,
        height: int | None = None,
        format: int = ...,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    @overload
    def Spline16(
        self,
        clip: _VideoNodeT,
        width: int | None = None,
        height: int | None = None,
        format: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> _VideoNodeT: ...
    @overload
    def Spline36(
        self,
        clip: _VideoNodeT,
        width: int | None = None,
        height: int | None = None,
        format: None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> _VideoNodeT: ...
    @overload
    def Spline36(
        self,
        clip: VideoNode,
        width: int | None = None,
        height: int | None = None,
        format: int = ...,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    @overload
    def Spline36(
        self,
        clip: _VideoNodeT,
        width: int | None = None,
        height: int | None = None,
        format: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> _VideoNodeT: ...
    @overload
    def Spline64(
        self,
        clip: _VideoNodeT,
        width: int | None = None,
        height: int | None = None,
        format: None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> _VideoNodeT: ...
    @overload
    def Spline64(
        self,
        clip: VideoNode,
        width: int | None = None,
        height: int | None = None,
        format: int = ...,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    @overload
    def Spline64(
        self,
        clip: _VideoNodeT,
        width: int | None = None,
        height: int | None = None,
        format: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> _VideoNodeT: ...
    def Bob(
        self,
        clip: VideoNode,
        filter: _DataType | None = None,
        tff: int | None = None,
        format: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    @overload
    def Custom(
        self,
        clip: _VideoNodeT,
        custom_kernel: _CustomKernelCallback,
        taps: int,
        width: int | None = None,
        height: int | None = None,
        format: None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> _VideoNodeT: ...
    @overload
    def Custom(
        self,
        clip: VideoNode,
        custom_kernel: _CustomKernelCallback,
        taps: int,
        width: int | None = None,
        height: int | None = None,
        format: int = ...,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    @overload
    def Custom(
        self,
        clip: _VideoNodeT,
        custom_kernel: _CustomKernelCallback,
        taps: int,
        width: int | None = None,
        height: int | None = None,
        format: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> _VideoNodeT: ...

class _Plugin_resize2_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "resize2" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    @overload
    def Bicubic(
        self,
        width: int | None = None,
        height: int | None = None,
        format: None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> VideoNode: ...
    @overload
    def Bicubic(
        self,
        width: int | None = None,
        height: int | None = None,
        format: int = ...,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    @overload
    def Bicubic(
        self,
        width: int | None = None,
        height: int | None = None,
        format: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> VideoNode: ...
    @overload
    def Bilinear(
        self,
        width: int | None = None,
        height: int | None = None,
        format: None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> VideoNode: ...
    @overload
    def Bilinear(
        self,
        width: int | None = None,
        height: int | None = None,
        format: int = ...,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    @overload
    def Bilinear(
        self,
        width: int | None = None,
        height: int | None = None,
        format: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> VideoNode: ...
    @overload
    def Lanczos(
        self,
        width: int | None = None,
        height: int | None = None,
        format: None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> VideoNode: ...
    @overload
    def Lanczos(
        self,
        width: int | None = None,
        height: int | None = None,
        format: int = ...,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    @overload
    def Lanczos(
        self,
        width: int | None = None,
        height: int | None = None,
        format: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> VideoNode: ...
    @overload
    def Point(
        self,
        width: int | None = None,
        height: int | None = None,
        format: None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> VideoNode: ...
    @overload
    def Point(
        self,
        width: int | None = None,
        height: int | None = None,
        format: int = ...,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    @overload
    def Point(
        self,
        width: int | None = None,
        height: int | None = None,
        format: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> VideoNode: ...
    @overload
    def Spline16(
        self,
        width: int | None = None,
        height: int | None = None,
        format: None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> VideoNode: ...
    @overload
    def Spline16(
        self,
        width: int | None = None,
        height: int | None = None,
        format: int = ...,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    @overload
    def Spline16(
        self,
        width: int | None = None,
        height: int | None = None,
        format: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> VideoNode: ...
    @overload
    def Spline36(
        self,
        width: int | None = None,
        height: int | None = None,
        format: None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> VideoNode: ...
    @overload
    def Spline36(
        self,
        width: int | None = None,
        height: int | None = None,
        format: int = ...,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    @overload
    def Spline36(
        self,
        width: int | None = None,
        height: int | None = None,
        format: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> VideoNode: ...
    @overload
    def Spline64(
        self,
        width: int | None = None,
        height: int | None = None,
        format: None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> VideoNode: ...
    @overload
    def Spline64(
        self,
        width: int | None = None,
        height: int | None = None,
        format: int = ...,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    @overload
    def Spline64(
        self,
        width: int | None = None,
        height: int | None = None,
        format: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> VideoNode: ...
    def Bob(
        self,
        filter: _DataType | None = None,
        tff: int | None = None,
        format: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        filter_param_a: float | None = None,
        filter_param_b: float | None = None,
        resample_filter_uv: _DataType | None = None,
        filter_param_a_uv: float | None = None,
        filter_param_b_uv: float | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    @overload
    def Custom(
        self,
        custom_kernel: _CustomKernelCallback,
        taps: int,
        width: int | None = None,
        height: int | None = None,
        format: None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> VideoNode: ...
    @overload
    def Custom(
        self,
        custom_kernel: _CustomKernelCallback,
        taps: int,
        width: int | None = None,
        height: int | None = None,
        format: int = ...,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    @overload
    def Custom(
        self,
        custom_kernel: _CustomKernelCallback,
        taps: int,
        width: int | None = None,
        height: int | None = None,
        format: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
        range_s: _DataType | None = None,
        chromaloc: int | None = None,
        chromaloc_s: _DataType | None = None,
        matrix_in: int | None = None,
        matrix_in_s: _DataType | None = None,
        transfer_in: int | None = None,
        transfer_in_s: _DataType | None = None,
        primaries_in: int | None = None,
        primaries_in_s: _DataType | None = None,
        range_in: int | None = None,
        range_in_s: _DataType | None = None,
        chromaloc_in: int | None = None,
        chromaloc_in_s: _DataType | None = None,
        dither_type: _DataType | None = None,
        cpu_type: _DataType | None = None,
        prefer_props: int | None = None,
        src_left: float | None = None,
        src_top: float | None = None,
        src_width: float | None = None,
        src_height: float | None = None,
        nominal_luminance: float | None = None,
        force: int | None = None,
        force_h: int | None = None,
        force_v: int | None = None,
        blur: float | None = None,
    ) -> VideoNode: ...

# end implementation

# implementation: sangnom

class _Plugin_sangnom_Core_Bound(Plugin):
    """This class implements the module definitions for the "sangnom" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def SangNom(
        self,
        clip: VideoNode,
        order: int | None = None,
        dh: int | None = None,
        aa: _SingleAndSequence[int] | None = None,
        planes: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...

class _Plugin_sangnom_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "sangnom" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def SangNom(
        self,
        order: int | None = None,
        dh: int | None = None,
        aa: _SingleAndSequence[int] | None = None,
        planes: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: scxvid

class _Plugin_scxvid_Core_Bound(Plugin):
    """This class implements the module definitions for the "scxvid" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Scxvid(
        self, clip: VideoNode, log: _DataType | None = None, use_slices: int | None = None
    ) -> ConstantFormatVideoNode: ...

class _Plugin_scxvid_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "scxvid" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def Scxvid(self, log: _DataType | None = None, use_slices: int | None = None) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: sneedif

_ReturnDict_sneedif_DeviceInfo = TypedDict(
    "_ReturnDict_sneedif_DeviceInfo",
    {
        "name": _DataType,
        "vendor": _DataType,
        "profile": _DataType,
        "version": _DataType,
        "max_compute_units": int,
        "max_work_group_size": int,
        "max_work_item_sizes": _SingleAndSequence[int],
        "image2D_max_width": int,
        "image2D_max_height": int,
        "image_support": int,
        "global_memory_cache_type": _DataType,
        "global_memory_cache": int,
        "global_memory_size": int,
        "max_constant_buffer_size": int,
        "max_constant_arguments": int,
        "local_memory_type": _DataType,
        "local_memory_size": int,
        "available": int,
        "compiler_available": int,
        "linker_available": int,
        "opencl_c_version": _DataType,
        "image_max_buffer_size": int,
    },
)
_ReturnDict_sneedif_ListDevices = TypedDict(
    "_ReturnDict_sneedif_ListDevices",
    {"numDevices": int, "deviceNames": _SingleAndSequence[_DataType], "platformNames": _SingleAndSequence[_DataType]},
)
_ReturnDict_sneedif_PlatformInfo = TypedDict(
    "_ReturnDict_sneedif_PlatformInfo",
    {"profile": _DataType, "version": _DataType, "name": _DataType, "vendor": _DataType},
)

class _Plugin_sneedif_Core_Bound(Plugin):
    """This class implements the module definitions for the "sneedif" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def DeviceInfo(self, device: int | None = None) -> _ReturnDict_sneedif_DeviceInfo: ...
    def ListDevices(self) -> _ReturnDict_sneedif_ListDevices: ...
    def NNEDI3(
        self,
        clip: VideoNode,
        field: int,
        dh: int | None = None,
        dw: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
        nsize: int | None = None,
        nns: int | None = None,
        qual: int | None = None,
        etype: int | None = None,
        pscrn: int | None = None,
        transpose_first: int | None = None,
        device: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def PlatformInfo(self, device: int | None = None) -> _ReturnDict_sneedif_PlatformInfo: ...

class _Plugin_sneedif_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "sneedif" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def NNEDI3(
        self,
        field: int,
        dh: int | None = None,
        dw: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
        nsize: int | None = None,
        nns: int | None = None,
        qual: int | None = None,
        etype: int | None = None,
        pscrn: int | None = None,
        transpose_first: int | None = None,
        device: int | None = None,
    ) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: std

class _Plugin_std_Core_Bound(Plugin):
    """This class implements the module definitions for the "std" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def AddBorders(
        self,
        clip: _VideoNodeT,
        left: int | None = None,
        right: int | None = None,
        top: int | None = None,
        bottom: int | None = None,
        color: _SingleAndSequence[float] | None = None,
    ) -> _VideoNodeT: ...
    def AssumeFPS(
        self, clip: _VideoNodeT, src: VideoNode | None = None, fpsnum: int | None = None, fpsden: int | None = None
    ) -> _VideoNodeT: ...
    def AssumeSampleRate(
        self, clip: AudioNode, src: AudioNode | None = None, samplerate: int | None = None
    ) -> AudioNode: ...
    def AudioGain(
        self, clip: AudioNode, gain: _SingleAndSequence[float] | None = None, overflow_error: int | None = None
    ) -> AudioNode: ...
    def AudioLoop(self, clip: AudioNode, times: int | None = None) -> AudioNode: ...
    def AudioMix(
        self,
        clips: _SingleAndSequence[AudioNode],
        matrix: _SingleAndSequence[float],
        channels_out: _SingleAndSequence[int],
        overflow_error: int | None = None,
    ) -> AudioNode: ...
    def AudioReverse(self, clip: AudioNode) -> AudioNode: ...
    def AudioSplice(self, clips: _SingleAndSequence[AudioNode]) -> AudioNode: ...
    def AudioTrim(
        self, clip: AudioNode, first: int | None = None, last: int | None = None, length: int | None = None
    ) -> AudioNode: ...
    def AverageFrames(
        self,
        clips: _SingleAndSequence[VideoNode],
        weights: _SingleAndSequence[float],
        scale: float | None = None,
        scenechange: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Binarize(
        self,
        clip: VideoNode,
        threshold: _SingleAndSequence[float] | None = None,
        v0: _SingleAndSequence[float] | None = None,
        v1: _SingleAndSequence[float] | None = None,
        planes: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...
    def BinarizeMask(
        self,
        clip: VideoNode,
        threshold: _SingleAndSequence[float] | None = None,
        v0: _SingleAndSequence[float] | None = None,
        v1: _SingleAndSequence[float] | None = None,
        planes: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...
    def BlankAudio(
        self,
        clip: AudioNode | None = None,
        channels: _SingleAndSequence[int] | None = None,
        bits: int | None = None,
        sampletype: int | None = None,
        samplerate: int | None = None,
        length: int | None = None,
        keep: int | None = None,
    ) -> AudioNode: ...
    @overload
    def BlankClip(
        self,
        clip: VideoNode | None = None,
        width: int | None = None,
        height: int | None = None,
        format: int | None = None,
        length: int | None = None,
        fpsnum: int | None = None,
        fpsden: int | None = None,
        color: _SingleAndSequence[float] | None = None,
        keep: int | None = None,
        varsize: int | None = None,
        varformat: None = None,
    ) -> ConstantFormatVideoNode: ...
    @overload
    def BlankClip(
        self,
        clip: VideoNode | None = None,
        width: int | None = None,
        height: int | None = None,
        format: int | None = None,
        length: int | None = None,
        fpsnum: int | None = None,
        fpsden: int | None = None,
        color: _SingleAndSequence[float] | None = None,
        keep: int | None = None,
        varsize: int | None = None,
        varformat: int = ...,
    ) -> VideoNode: ...
    @overload
    def BlankClip(
        self,
        clip: _VideoNodeT | None = None,
        width: int | None = None,
        height: int | None = None,
        format: int | None = None,
        length: int | None = None,
        fpsnum: int | None = None,
        fpsden: int | None = None,
        color: _SingleAndSequence[float] | None = None,
        keep: int | None = None,
        varsize: int | None = None,
        varformat: int | None = None,
    ) -> _VideoNodeT: ...
    def BoxBlur(
        self,
        clip: VideoNode,
        planes: _SingleAndSequence[int] | None = None,
        hradius: int | None = None,
        hpasses: int | None = None,
        vradius: int | None = None,
        vpasses: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Cache(
        self, clip: VideoNode, size: int | None = None, fixed: int | None = None, make_linear: int | None = None
    ) -> ConstantFormatVideoNode: ...
    def ClipToProp(
        self, clip: VideoNode, mclip: VideoNode, prop: _DataType | None = None
    ) -> ConstantFormatVideoNode: ...
    def Convolution(
        self,
        clip: VideoNode,
        matrix: _SingleAndSequence[float],
        bias: float | None = None,
        divisor: float | None = None,
        planes: _SingleAndSequence[int] | None = None,
        saturate: int | None = None,
        mode: _DataType | None = None,
    ) -> ConstantFormatVideoNode: ...
    def CopyFrameProps(
        self, clip: _VideoNodeT, prop_src: VideoNode, props: _SingleAndSequence[_DataType] | None = None
    ) -> _VideoNodeT: ...
    def Crop(
        self,
        clip: VideoNode,
        left: int | None = None,
        right: int | None = None,
        top: int | None = None,
        bottom: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def CropAbs(
        self,
        clip: _VideoNodeT,
        width: int,
        height: int,
        left: int | None = None,
        top: int | None = None,
        x: int | None = None,
        y: int | None = None,
    ) -> _VideoNodeT: ...
    def CropRel(
        self,
        clip: VideoNode,
        left: int | None = None,
        right: int | None = None,
        top: int | None = None,
        bottom: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Deflate(
        self, clip: VideoNode, planes: _SingleAndSequence[int] | None = None, threshold: float | None = None
    ) -> ConstantFormatVideoNode: ...
    def DeleteFrames(self, clip: _VideoNodeT, frames: _SingleAndSequence[int]) -> _VideoNodeT: ...
    def DoubleWeave(self, clip: VideoNode, tff: int | None = None) -> ConstantFormatVideoNode: ...
    def DuplicateFrames(self, clip: _VideoNodeT, frames: _SingleAndSequence[int]) -> _VideoNodeT: ...
    def Expr(
        self, clips: _SingleAndSequence[VideoNode], expr: _SingleAndSequence[_DataType], format: int | None = None
    ) -> ConstantFormatVideoNode: ...
    def FlipHorizontal(self, clip: _VideoNodeT) -> _VideoNodeT: ...
    def FlipVertical(self, clip: _VideoNodeT) -> _VideoNodeT: ...
    def FrameEval(
        self,
        clip: _VideoNodeT,
        eval: _VSMapValueCallback[_VapourSynthMapValue],
        prop_src: _SingleAndSequence[VideoNode] | None = None,
        clip_src: _SingleAndSequence[VideoNode] | None = None,
    ) -> _VideoNodeT: ...
    def FreezeFrames(
        self,
        clip: _VideoNodeT,
        first: _SingleAndSequence[int] | None = None,
        last: _SingleAndSequence[int] | None = None,
        replacement: _SingleAndSequence[int] | None = None,
    ) -> _VideoNodeT: ...
    def Inflate(
        self, clip: VideoNode, planes: _SingleAndSequence[int] | None = None, threshold: float | None = None
    ) -> ConstantFormatVideoNode: ...
    @overload
    def Interleave(
        self,
        clips: _SingleAndSequence[VideoNode],
        extend: int | None = None,
        mismatch: None = None,
        modify_duration: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    @overload
    def Interleave(
        self,
        clips: _SingleAndSequence[VideoNode],
        extend: int | None = None,
        mismatch: int = ...,
        modify_duration: int | None = None,
    ) -> VideoNode: ...
    @overload
    def Interleave(
        self,
        clips: _SingleAndSequence[VideoNode],
        extend: int | None = None,
        mismatch: int | None = None,
        modify_duration: int | None = None,
    ) -> VideoNode: ...
    def Invert(self, clip: _VideoNodeT, planes: _SingleAndSequence[int] | None = None) -> _VideoNodeT: ...
    def InvertMask(self, clip: _VideoNodeT, planes: _SingleAndSequence[int] | None = None) -> _VideoNodeT: ...
    def Levels(
        self,
        clip: VideoNode,
        min_in: _SingleAndSequence[float] | None = None,
        max_in: _SingleAndSequence[float] | None = None,
        gamma: _SingleAndSequence[float] | None = None,
        min_out: _SingleAndSequence[float] | None = None,
        max_out: _SingleAndSequence[float] | None = None,
        planes: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Limiter(
        self,
        clip: VideoNode,
        min: _SingleAndSequence[float] | None = None,
        max: _SingleAndSequence[float] | None = None,
        planes: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...
    def LoadAllPlugins(self, path: _DataType) -> None: ...
    def LoadPlugin(
        self,
        path: _DataType,
        altsearchpath: int | None = None,
        forcens: _DataType | None = None,
        forceid: _DataType | None = None,
    ) -> None: ...
    def Loop(self, clip: _VideoNodeT, times: int | None = None) -> _VideoNodeT: ...
    def Lut(
        self,
        clip: VideoNode,
        planes: _SingleAndSequence[int] | None = None,
        lut: _SingleAndSequence[int] | None = None,
        lutf: _SingleAndSequence[float] | None = None,
        function: _VSMapValueCallback[_VapourSynthMapValue] | None = None,
        bits: int | None = None,
        floatout: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Lut2(
        self,
        clipa: VideoNode,
        clipb: VideoNode,
        planes: _SingleAndSequence[int] | None = None,
        lut: _SingleAndSequence[int] | None = None,
        lutf: _SingleAndSequence[float] | None = None,
        function: _VSMapValueCallback[_VapourSynthMapValue] | None = None,
        bits: int | None = None,
        floatout: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def MakeDiff(
        self, clipa: VideoNode, clipb: VideoNode, planes: _SingleAndSequence[int] | None = None
    ) -> ConstantFormatVideoNode: ...
    def MakeFullDiff(self, clipa: VideoNode, clipb: VideoNode) -> ConstantFormatVideoNode: ...
    def MaskedMerge(
        self,
        clipa: VideoNode,
        clipb: VideoNode,
        mask: VideoNode,
        planes: _SingleAndSequence[int] | None = None,
        first_plane: int | None = None,
        premultiplied: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Maximum(
        self,
        clip: VideoNode,
        planes: _SingleAndSequence[int] | None = None,
        threshold: float | None = None,
        coordinates: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Median(self, clip: VideoNode, planes: _SingleAndSequence[int] | None = None) -> ConstantFormatVideoNode: ...
    def Merge(
        self, clipa: VideoNode, clipb: VideoNode, weight: _SingleAndSequence[float] | None = None
    ) -> ConstantFormatVideoNode: ...
    def MergeDiff(
        self, clipa: VideoNode, clipb: VideoNode, planes: _SingleAndSequence[int] | None = None
    ) -> ConstantFormatVideoNode: ...
    def MergeFullDiff(self, clipa: VideoNode, clipb: VideoNode) -> ConstantFormatVideoNode: ...
    def Minimum(
        self,
        clip: VideoNode,
        planes: _SingleAndSequence[int] | None = None,
        threshold: float | None = None,
        coordinates: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...
    def ModifyFrame(
        self,
        clip: _VideoNodeT,
        clips: _SingleAndSequence[VideoNode],
        selector: _VSMapValueCallback[_VapourSynthMapValue],
    ) -> _VideoNodeT: ...
    def PEMVerifier(
        self,
        clip: VideoNode,
        upper: _SingleAndSequence[float] | None = None,
        lower: _SingleAndSequence[float] | None = None,
    ) -> ConstantFormatVideoNode: ...
    def PlaneStats(
        self, clipa: VideoNode, clipb: VideoNode | None = None, plane: int | None = None, prop: _DataType | None = None
    ) -> ConstantFormatVideoNode: ...
    def PreMultiply(self, clip: VideoNode, alpha: VideoNode) -> ConstantFormatVideoNode: ...
    def Prewitt(
        self, clip: VideoNode, planes: _SingleAndSequence[int] | None = None, scale: float | None = None
    ) -> ConstantFormatVideoNode: ...
    def PropToClip(self, clip: VideoNode, prop: _DataType | None = None) -> ConstantFormatVideoNode: ...
    def RemoveFrameProps(
        self, clip: _VideoNodeT, props: _SingleAndSequence[_DataType] | None = None
    ) -> _VideoNodeT: ...
    def Reverse(self, clip: _VideoNodeT) -> _VideoNodeT: ...
    def SelectEvery(
        self, clip: _VideoNodeT, cycle: int, offsets: _SingleAndSequence[int], modify_duration: int | None = None
    ) -> _VideoNodeT: ...
    def SeparateFields(
        self, clip: VideoNode, tff: int | None = None, modify_duration: int | None = None
    ) -> ConstantFormatVideoNode: ...
    def SetAudioCache(
        self,
        clip: AudioNode,
        mode: int | None = None,
        fixedsize: int | None = None,
        maxsize: int | None = None,
        maxhistory: int | None = None,
    ) -> None: ...
    def SetFieldBased(self, clip: _VideoNodeT, value: int) -> _VideoNodeT: ...
    def SetFrameProp(
        self,
        clip: _VideoNodeT,
        prop: _DataType,
        intval: _SingleAndSequence[int] | None = None,
        floatval: _SingleAndSequence[float] | None = None,
        data: _SingleAndSequence[_DataType] | None = None,
    ) -> _VideoNodeT: ...
    def SetFrameProps(self, clip: _VideoNodeT, **kwargs: _VapourSynthMapValue) -> _VideoNodeT: ...
    def SetMaxCPU(self, cpu: _DataType) -> ConstantFormatVideoNode: ...
    def SetVideoCache(
        self,
        clip: VideoNode,
        mode: int | None = None,
        fixedsize: int | None = None,
        maxsize: int | None = None,
        maxhistory: int | None = None,
    ) -> None: ...
    def ShuffleChannels(
        self,
        clips: _SingleAndSequence[AudioNode],
        channels_in: _SingleAndSequence[int],
        channels_out: _SingleAndSequence[int],
    ) -> AudioNode: ...
    def ShufflePlanes(
        self,
        clips: _SingleAndSequence[VideoNode],
        planes: _SingleAndSequence[int],
        colorfamily: int,
        prop_src: VideoNode | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Sobel(
        self, clip: VideoNode, planes: _SingleAndSequence[int] | None = None, scale: float | None = None
    ) -> ConstantFormatVideoNode: ...
    @overload
    def Splice(self, clips: _SingleAndSequence[VideoNode], mismatch: None = None) -> ConstantFormatVideoNode: ...
    @overload
    def Splice(self, clips: _SingleAndSequence[VideoNode], mismatch: int = ...) -> VideoNode: ...
    @overload
    def Splice(self, clips: _SingleAndSequence[VideoNode], mismatch: int | None = None) -> VideoNode: ...
    def SplitChannels(self, clip: AudioNode) -> _SingleAndSequence[AudioNode]: ...
    def SplitPlanes(self, clip: VideoNode) -> _SingleAndSequence[ConstantFormatVideoNode]: ...
    def StackHorizontal(self, clips: _SingleAndSequence[VideoNode]) -> ConstantFormatVideoNode: ...
    def StackVertical(self, clips: _SingleAndSequence[VideoNode]) -> ConstantFormatVideoNode: ...
    def TestAudio(
        self,
        channels: _SingleAndSequence[int] | None = None,
        bits: int | None = None,
        isfloat: int | None = None,
        samplerate: int | None = None,
        length: int | None = None,
    ) -> AudioNode: ...
    def Transpose(self, clip: VideoNode) -> ConstantFormatVideoNode: ...
    def Trim(
        self, clip: _VideoNodeT, first: int | None = None, last: int | None = None, length: int | None = None
    ) -> _VideoNodeT: ...
    def Turn180(self, clip: _VideoNodeT) -> _VideoNodeT: ...

class _Plugin_std_VideoNode_Bound(Plugin, Generic[_VideoNodeT]):
    """This class implements the module definitions for the "std" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def AddBorders(
        self,
        left: int | None = None,
        right: int | None = None,
        top: int | None = None,
        bottom: int | None = None,
        color: _SingleAndSequence[float] | None = None,
    ) -> _VideoNodeT: ...
    def AssumeFPS(
        self, src: VideoNode | None = None, fpsnum: int | None = None, fpsden: int | None = None
    ) -> _VideoNodeT: ...
    def AverageFrames(
        self,
        weights: _SingleAndSequence[float],
        scale: float | None = None,
        scenechange: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Binarize(
        self,
        threshold: _SingleAndSequence[float] | None = None,
        v0: _SingleAndSequence[float] | None = None,
        v1: _SingleAndSequence[float] | None = None,
        planes: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...
    def BinarizeMask(
        self,
        threshold: _SingleAndSequence[float] | None = None,
        v0: _SingleAndSequence[float] | None = None,
        v1: _SingleAndSequence[float] | None = None,
        planes: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...
    @overload
    def BlankClip(
        self,
        width: int | None = None,
        height: int | None = None,
        format: int | None = None,
        length: int | None = None,
        fpsnum: int | None = None,
        fpsden: int | None = None,
        color: _SingleAndSequence[float] | None = None,
        keep: int | None = None,
        varsize: int | None = None,
        varformat: None = None,
    ) -> _VideoNodeT: ...
    @overload
    def BlankClip(
        self,
        width: int | None = None,
        height: int | None = None,
        format: int | None = None,
        length: int | None = None,
        fpsnum: int | None = None,
        fpsden: int | None = None,
        color: _SingleAndSequence[float] | None = None,
        keep: int | None = None,
        varsize: int | None = None,
        varformat: int = ...,
    ) -> VideoNode: ...
    @overload
    def BlankClip(
        self,
        width: int | None = None,
        height: int | None = None,
        format: int | None = None,
        length: int | None = None,
        fpsnum: int | None = None,
        fpsden: int | None = None,
        color: _SingleAndSequence[float] | None = None,
        keep: int | None = None,
        varsize: int | None = None,
        varformat: int | None = None,
    ) -> _VideoNodeT: ...
    def BoxBlur(
        self,
        planes: _SingleAndSequence[int] | None = None,
        hradius: int | None = None,
        hpasses: int | None = None,
        vradius: int | None = None,
        vpasses: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Cache(
        self, size: int | None = None, fixed: int | None = None, make_linear: int | None = None
    ) -> ConstantFormatVideoNode: ...
    def ClipToProp(self, mclip: VideoNode, prop: _DataType | None = None) -> ConstantFormatVideoNode: ...
    def Convolution(
        self,
        matrix: _SingleAndSequence[float],
        bias: float | None = None,
        divisor: float | None = None,
        planes: _SingleAndSequence[int] | None = None,
        saturate: int | None = None,
        mode: _DataType | None = None,
    ) -> ConstantFormatVideoNode: ...
    def CopyFrameProps(
        self, prop_src: VideoNode, props: _SingleAndSequence[_DataType] | None = None
    ) -> _VideoNodeT: ...
    def Crop(
        self, left: int | None = None, right: int | None = None, top: int | None = None, bottom: int | None = None
    ) -> ConstantFormatVideoNode: ...
    def CropAbs(
        self,
        width: int,
        height: int,
        left: int | None = None,
        top: int | None = None,
        x: int | None = None,
        y: int | None = None,
    ) -> _VideoNodeT: ...
    def CropRel(
        self, left: int | None = None, right: int | None = None, top: int | None = None, bottom: int | None = None
    ) -> ConstantFormatVideoNode: ...
    def Deflate(
        self, planes: _SingleAndSequence[int] | None = None, threshold: float | None = None
    ) -> ConstantFormatVideoNode: ...
    def DeleteFrames(self, frames: _SingleAndSequence[int]) -> _VideoNodeT: ...
    def DoubleWeave(self, tff: int | None = None) -> ConstantFormatVideoNode: ...
    def DuplicateFrames(self, frames: _SingleAndSequence[int]) -> _VideoNodeT: ...
    def Expr(self, expr: _SingleAndSequence[_DataType], format: int | None = None) -> ConstantFormatVideoNode: ...
    def FlipHorizontal(self) -> _VideoNodeT: ...
    def FlipVertical(self) -> _VideoNodeT: ...
    def FrameEval(
        self,
        eval: _VSMapValueCallback[_VapourSynthMapValue],
        prop_src: _SingleAndSequence[VideoNode] | None = None,
        clip_src: _SingleAndSequence[VideoNode] | None = None,
    ) -> _VideoNodeT: ...
    def FreezeFrames(
        self,
        first: _SingleAndSequence[int] | None = None,
        last: _SingleAndSequence[int] | None = None,
        replacement: _SingleAndSequence[int] | None = None,
    ) -> _VideoNodeT: ...
    def Inflate(
        self, planes: _SingleAndSequence[int] | None = None, threshold: float | None = None
    ) -> ConstantFormatVideoNode: ...
    @overload
    def Interleave(
        self, extend: int | None = None, mismatch: int | None = None, modify_duration: None = None
    ) -> _VideoNodeT: ...
    @overload
    def Interleave(
        self, extend: int | None = None, mismatch: int | None = None, modify_duration: int = ...
    ) -> VideoNode: ...
    @overload
    def Interleave(
        self, extend: int | None = None, mismatch: int | None = None, modify_duration: int | None = None
    ) -> _VideoNodeT: ...
    def Invert(self, planes: _SingleAndSequence[int] | None = None) -> _VideoNodeT: ...
    def InvertMask(self, planes: _SingleAndSequence[int] | None = None) -> _VideoNodeT: ...
    def Levels(
        self,
        min_in: _SingleAndSequence[float] | None = None,
        max_in: _SingleAndSequence[float] | None = None,
        gamma: _SingleAndSequence[float] | None = None,
        min_out: _SingleAndSequence[float] | None = None,
        max_out: _SingleAndSequence[float] | None = None,
        planes: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Limiter(
        self,
        min: _SingleAndSequence[float] | None = None,
        max: _SingleAndSequence[float] | None = None,
        planes: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Loop(self, times: int | None = None) -> _VideoNodeT: ...
    def Lut(
        self,
        planes: _SingleAndSequence[int] | None = None,
        lut: _SingleAndSequence[int] | None = None,
        lutf: _SingleAndSequence[float] | None = None,
        function: _VSMapValueCallback[_VapourSynthMapValue] | None = None,
        bits: int | None = None,
        floatout: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Lut2(
        self,
        clipb: VideoNode,
        planes: _SingleAndSequence[int] | None = None,
        lut: _SingleAndSequence[int] | None = None,
        lutf: _SingleAndSequence[float] | None = None,
        function: _VSMapValueCallback[_VapourSynthMapValue] | None = None,
        bits: int | None = None,
        floatout: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def MakeDiff(self, clipb: VideoNode, planes: _SingleAndSequence[int] | None = None) -> ConstantFormatVideoNode: ...
    def MakeFullDiff(self, clipb: VideoNode) -> ConstantFormatVideoNode: ...
    def MaskedMerge(
        self,
        clipb: VideoNode,
        mask: VideoNode,
        planes: _SingleAndSequence[int] | None = None,
        first_plane: int | None = None,
        premultiplied: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Maximum(
        self,
        planes: _SingleAndSequence[int] | None = None,
        threshold: float | None = None,
        coordinates: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Median(self, planes: _SingleAndSequence[int] | None = None) -> ConstantFormatVideoNode: ...
    def Merge(self, clipb: VideoNode, weight: _SingleAndSequence[float] | None = None) -> ConstantFormatVideoNode: ...
    def MergeDiff(self, clipb: VideoNode, planes: _SingleAndSequence[int] | None = None) -> ConstantFormatVideoNode: ...
    def MergeFullDiff(self, clipb: VideoNode) -> ConstantFormatVideoNode: ...
    def Minimum(
        self,
        planes: _SingleAndSequence[int] | None = None,
        threshold: float | None = None,
        coordinates: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...
    def ModifyFrame(
        self, clips: _SingleAndSequence[VideoNode], selector: _VSMapValueCallback[_VapourSynthMapValue]
    ) -> _VideoNodeT: ...
    def PEMVerifier(
        self, upper: _SingleAndSequence[float] | None = None, lower: _SingleAndSequence[float] | None = None
    ) -> ConstantFormatVideoNode: ...
    def PlaneStats(
        self, clipb: VideoNode | None = None, plane: int | None = None, prop: _DataType | None = None
    ) -> ConstantFormatVideoNode: ...
    def PreMultiply(self, alpha: VideoNode) -> ConstantFormatVideoNode: ...
    def Prewitt(
        self, planes: _SingleAndSequence[int] | None = None, scale: float | None = None
    ) -> ConstantFormatVideoNode: ...
    def PropToClip(self, prop: _DataType | None = None) -> ConstantFormatVideoNode: ...
    def RemoveFrameProps(self, props: _SingleAndSequence[_DataType] | None = None) -> _VideoNodeT: ...
    def Reverse(self) -> _VideoNodeT: ...
    def SelectEvery(
        self, cycle: int, offsets: _SingleAndSequence[int], modify_duration: int | None = None
    ) -> _VideoNodeT: ...
    def SeparateFields(self, tff: int | None = None, modify_duration: int | None = None) -> ConstantFormatVideoNode: ...
    def SetFieldBased(self, value: int) -> _VideoNodeT: ...
    def SetFrameProp(
        self,
        prop: _DataType,
        intval: _SingleAndSequence[int] | None = None,
        floatval: _SingleAndSequence[float] | None = None,
        data: _SingleAndSequence[_DataType] | None = None,
    ) -> _VideoNodeT: ...
    def SetFrameProps(self, **kwargs: Any) -> _VideoNodeT: ...
    def SetVideoCache(
        self,
        mode: int | None = None,
        fixedsize: int | None = None,
        maxsize: int | None = None,
        maxhistory: int | None = None,
    ) -> None: ...
    def ShufflePlanes(
        self, planes: _SingleAndSequence[int], colorfamily: int, prop_src: VideoNode | None = None
    ) -> ConstantFormatVideoNode: ...
    def Sobel(
        self, planes: _SingleAndSequence[int] | None = None, scale: float | None = None
    ) -> ConstantFormatVideoNode: ...
    def Splice(self, mismatch: int | None = None) -> _VideoNodeT: ...
    def SplitPlanes(self) -> _SingleAndSequence[ConstantFormatVideoNode]: ...
    def StackHorizontal(self) -> _VideoNodeT: ...
    def StackVertical(self) -> _VideoNodeT: ...
    def Transpose(self) -> ConstantFormatVideoNode: ...
    def Trim(self, first: int | None = None, last: int | None = None, length: int | None = None) -> _VideoNodeT: ...
    def Turn180(self) -> _VideoNodeT: ...

class _Plugin_std_AudioNode_Bound(Plugin):
    """This class implements the module definitions for the "std" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def AssumeSampleRate(self, src: AudioNode | None = None, samplerate: int | None = None) -> AudioNode: ...
    def AudioGain(
        self, gain: _SingleAndSequence[float] | None = None, overflow_error: int | None = None
    ) -> AudioNode: ...
    def AudioLoop(self, times: int | None = None) -> AudioNode: ...
    def AudioMix(
        self,
        matrix: _SingleAndSequence[float],
        channels_out: _SingleAndSequence[int],
        overflow_error: int | None = None,
    ) -> AudioNode: ...
    def AudioReverse(self) -> AudioNode: ...
    def AudioSplice(self) -> AudioNode: ...
    def AudioTrim(self, first: int | None = None, last: int | None = None, length: int | None = None) -> AudioNode: ...
    def BlankAudio(
        self,
        channels: _SingleAndSequence[int] | None = None,
        bits: int | None = None,
        sampletype: int | None = None,
        samplerate: int | None = None,
        length: int | None = None,
        keep: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def SetAudioCache(
        self,
        mode: int | None = None,
        fixedsize: int | None = None,
        maxsize: int | None = None,
        maxhistory: int | None = None,
    ) -> None: ...
    def ShuffleChannels(
        self, channels_in: _SingleAndSequence[int], channels_out: _SingleAndSequence[int]
    ) -> AudioNode: ...
    def SplitChannels(self) -> AudioNode: ...

# end implementation

# implementation: sub

class _Plugin_sub_Core_Bound(Plugin):
    """This class implements the module definitions for the "sub" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def ImageFile(
        self,
        clip: VideoNode,
        file: _DataType,
        id: int | None = None,
        palette: _SingleAndSequence[int] | None = None,
        gray: int | None = None,
        info: int | None = None,
        flatten: int | None = None,
        blend: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Subtitle(
        self,
        clip: VideoNode,
        text: _DataType,
        start: int | None = None,
        end: int | None = None,
        debuglevel: int | None = None,
        fontdir: _DataType | None = None,
        linespacing: float | None = None,
        margins: _SingleAndSequence[int] | None = None,
        sar: float | None = None,
        style: _DataType | None = None,
        blend: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def TextFile(
        self,
        clip: VideoNode,
        file: _DataType,
        charset: _DataType | None = None,
        scale: float | None = None,
        debuglevel: int | None = None,
        fontdir: _DataType | None = None,
        linespacing: float | None = None,
        margins: _SingleAndSequence[int] | None = None,
        sar: float | None = None,
        style: _DataType | None = None,
        blend: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
    ) -> ConstantFormatVideoNode: ...

class _Plugin_sub_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "sub" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def ImageFile(
        self,
        file: _DataType,
        id: int | None = None,
        palette: _SingleAndSequence[int] | None = None,
        gray: int | None = None,
        info: int | None = None,
        flatten: int | None = None,
        blend: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Subtitle(
        self,
        text: _DataType,
        start: int | None = None,
        end: int | None = None,
        debuglevel: int | None = None,
        fontdir: _DataType | None = None,
        linespacing: float | None = None,
        margins: _SingleAndSequence[int] | None = None,
        sar: float | None = None,
        style: _DataType | None = None,
        blend: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def TextFile(
        self,
        file: _DataType,
        charset: _DataType | None = None,
        scale: float | None = None,
        debuglevel: int | None = None,
        fontdir: _DataType | None = None,
        linespacing: float | None = None,
        margins: _SingleAndSequence[int] | None = None,
        sar: float | None = None,
        style: _DataType | None = None,
        blend: int | None = None,
        matrix: int | None = None,
        matrix_s: _DataType | None = None,
        transfer: int | None = None,
        transfer_s: _DataType | None = None,
        primaries: int | None = None,
        primaries_s: _DataType | None = None,
        range: int | None = None,
    ) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: tcanny

class _Plugin_tcanny_Core_Bound(Plugin):
    """This class implements the module definitions for the "tcanny" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def TCanny(
        self,
        clip: VideoNode,
        sigma: _SingleAndSequence[float] | None = None,
        sigma_v: _SingleAndSequence[float] | None = None,
        t_h: float | None = None,
        t_l: float | None = None,
        mode: int | None = None,
        op: int | None = None,
        scale: float | None = None,
        opt: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...

class _Plugin_tcanny_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "tcanny" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def TCanny(
        self,
        sigma: _SingleAndSequence[float] | None = None,
        sigma_v: _SingleAndSequence[float] | None = None,
        t_h: float | None = None,
        t_l: float | None = None,
        mode: int | None = None,
        op: int | None = None,
        scale: float | None = None,
        opt: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: tedgemask

class _Plugin_tedgemask_Core_Bound(Plugin):
    """This class implements the module definitions for the "tedgemask" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def TEdgeMask(
        self,
        clip: VideoNode,
        threshold: _SingleAndSequence[float] | None = None,
        type: int | None = None,
        link: int | None = None,
        scale: float | None = None,
        planes: _SingleAndSequence[int] | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...

class _Plugin_tedgemask_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "tedgemask" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def TEdgeMask(
        self,
        threshold: _SingleAndSequence[float] | None = None,
        type: int | None = None,
        link: int | None = None,
        scale: float | None = None,
        planes: _SingleAndSequence[int] | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: text

class _Plugin_text_Core_Bound(Plugin):
    """This class implements the module definitions for the "text" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def ClipInfo(
        self, clip: VideoNode, alignment: int | None = None, scale: int | None = None
    ) -> ConstantFormatVideoNode: ...
    def CoreInfo(
        self, clip: VideoNode | None = None, alignment: int | None = None, scale: int | None = None
    ) -> ConstantFormatVideoNode: ...
    def FrameNum(
        self, clip: VideoNode, alignment: int | None = None, scale: int | None = None
    ) -> ConstantFormatVideoNode: ...
    def FrameProps(
        self,
        clip: VideoNode,
        props: _SingleAndSequence[_DataType] | None = None,
        alignment: int | None = None,
        scale: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Text(
        self, clip: VideoNode, text: _DataType, alignment: int | None = None, scale: int | None = None
    ) -> ConstantFormatVideoNode: ...

class _Plugin_text_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "text" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def ClipInfo(self, alignment: int | None = None, scale: int | None = None) -> ConstantFormatVideoNode: ...
    def CoreInfo(self, alignment: int | None = None, scale: int | None = None) -> ConstantFormatVideoNode: ...
    def FrameNum(self, alignment: int | None = None, scale: int | None = None) -> ConstantFormatVideoNode: ...
    def FrameProps(
        self, props: _SingleAndSequence[_DataType] | None = None, alignment: int | None = None, scale: int | None = None
    ) -> ConstantFormatVideoNode: ...
    def Text(
        self, text: _DataType, alignment: int | None = None, scale: int | None = None
    ) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: vivtc

class _Plugin_vivtc_Core_Bound(Plugin):
    """This class implements the module definitions for the "vivtc" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def VDecimate(
        self,
        clip: VideoNode,
        cycle: int | None = None,
        chroma: int | None = None,
        dupthresh: float | None = None,
        scthresh: float | None = None,
        blockx: int | None = None,
        blocky: int | None = None,
        clip2: VideoNode | None = None,
        ovr: _DataType | None = None,
        dryrun: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def VFM(
        self,
        clip: VideoNode,
        order: int,
        field: int | None = None,
        mode: int | None = None,
        mchroma: int | None = None,
        cthresh: int | None = None,
        mi: int | None = None,
        chroma: int | None = None,
        blockx: int | None = None,
        blocky: int | None = None,
        y0: int | None = None,
        y1: int | None = None,
        scthresh: float | None = None,
        micmatch: int | None = None,
        micout: int | None = None,
        clip2: VideoNode | None = None,
    ) -> ConstantFormatVideoNode: ...

class _Plugin_vivtc_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "vivtc" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def VDecimate(
        self,
        cycle: int | None = None,
        chroma: int | None = None,
        dupthresh: float | None = None,
        scthresh: float | None = None,
        blockx: int | None = None,
        blocky: int | None = None,
        clip2: VideoNode | None = None,
        ovr: _DataType | None = None,
        dryrun: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def VFM(
        self,
        order: int,
        field: int | None = None,
        mode: int | None = None,
        mchroma: int | None = None,
        cthresh: int | None = None,
        mi: int | None = None,
        chroma: int | None = None,
        blockx: int | None = None,
        blocky: int | None = None,
        y0: int | None = None,
        y1: int | None = None,
        scthresh: float | None = None,
        micmatch: int | None = None,
        micout: int | None = None,
        clip2: VideoNode | None = None,
    ) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: vszip

class _Plugin_vszip_Core_Bound(Plugin):
    """This class implements the module definitions for the "vszip" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def AdaptiveBinarize(self, clip: VideoNode, clip2: VideoNode, c: int | None = None) -> ConstantFormatVideoNode: ...
    def Bilateral(
        self,
        clip: VideoNode,
        ref: VideoNode | None = None,
        sigmaS: _SingleAndSequence[float] | None = None,
        sigmaR: _SingleAndSequence[float] | None = None,
        planes: _SingleAndSequence[int] | None = None,
        algorithm: _SingleAndSequence[int] | None = None,
        PBFICnum: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...
    def BoxBlur(
        self,
        clip: VideoNode,
        planes: _SingleAndSequence[int] | None = None,
        hradius: int | None = None,
        hpasses: int | None = None,
        vradius: int | None = None,
        vpasses: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Checkmate(
        self, clip: VideoNode, thr: int | None = None, tmax: int | None = None, tthr2: int | None = None
    ) -> ConstantFormatVideoNode: ...
    def CLAHE(
        self, clip: VideoNode, limit: int | None = None, tiles: _SingleAndSequence[int] | None = None
    ) -> ConstantFormatVideoNode: ...
    def ColorMap(self, clip: VideoNode, color: int | None = None) -> ConstantFormatVideoNode: ...
    def CombMaskMT(
        self, clip: VideoNode, thY1: int | None = None, thY2: int | None = None
    ) -> ConstantFormatVideoNode: ...
    def ImageRead(
        self, path: _SingleAndSequence[_DataType], validate: int | None = None
    ) -> ConstantFormatVideoNode: ...
    def Limiter(
        self,
        clip: VideoNode,
        min: _SingleAndSequence[float] | None = None,
        max: _SingleAndSequence[float] | None = None,
        tv_range: int | None = None,
        mask: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...
    def LimitFilter(
        self,
        flt: VideoNode,
        src: VideoNode,
        ref: VideoNode | None = None,
        dark_thr: _SingleAndSequence[float] | None = None,
        bright_thr: _SingleAndSequence[float] | None = None,
        elast: _SingleAndSequence[float] | None = None,
        planes: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Metrics(
        self, reference: VideoNode, distorted: VideoNode, mode: int | None = None
    ) -> ConstantFormatVideoNode: ...
    def PackRGB(self, clip: VideoNode) -> ConstantFormatVideoNode: ...
    def PlaneAverage(
        self,
        clipa: VideoNode,
        exclude: _SingleAndSequence[int],
        clipb: VideoNode | None = None,
        planes: _SingleAndSequence[int] | None = None,
        prop: _DataType | None = None,
    ) -> ConstantFormatVideoNode: ...
    def PlaneMinMax(
        self,
        clipa: VideoNode,
        minthr: float | None = None,
        maxthr: float | None = None,
        clipb: VideoNode | None = None,
        planes: _SingleAndSequence[int] | None = None,
        prop: _DataType | None = None,
    ) -> ConstantFormatVideoNode: ...
    def RFS(
        self,
        clipa: VideoNode,
        clipb: VideoNode,
        frames: _SingleAndSequence[int],
        mismatch: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...
    def SSIMULACRA2(self, reference: VideoNode, distorted: VideoNode) -> ConstantFormatVideoNode: ...
    def XPSNR(
        self, reference: VideoNode, distorted: VideoNode, temporal: int | None = None, verbose: int | None = None
    ) -> ConstantFormatVideoNode: ...

class _Plugin_vszip_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "vszip" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def AdaptiveBinarize(self, clip2: VideoNode, c: int | None = None) -> ConstantFormatVideoNode: ...
    def Bilateral(
        self,
        ref: VideoNode | None = None,
        sigmaS: _SingleAndSequence[float] | None = None,
        sigmaR: _SingleAndSequence[float] | None = None,
        planes: _SingleAndSequence[int] | None = None,
        algorithm: _SingleAndSequence[int] | None = None,
        PBFICnum: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...
    def BoxBlur(
        self,
        planes: _SingleAndSequence[int] | None = None,
        hradius: int | None = None,
        hpasses: int | None = None,
        vradius: int | None = None,
        vpasses: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Checkmate(
        self, thr: int | None = None, tmax: int | None = None, tthr2: int | None = None
    ) -> ConstantFormatVideoNode: ...
    def CLAHE(
        self, limit: int | None = None, tiles: _SingleAndSequence[int] | None = None
    ) -> ConstantFormatVideoNode: ...
    def ColorMap(self, color: int | None = None) -> ConstantFormatVideoNode: ...
    def CombMaskMT(self, thY1: int | None = None, thY2: int | None = None) -> ConstantFormatVideoNode: ...
    def Limiter(
        self,
        min: _SingleAndSequence[float] | None = None,
        max: _SingleAndSequence[float] | None = None,
        tv_range: int | None = None,
        mask: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...
    def LimitFilter(
        self,
        src: VideoNode,
        ref: VideoNode | None = None,
        dark_thr: _SingleAndSequence[float] | None = None,
        bright_thr: _SingleAndSequence[float] | None = None,
        elast: _SingleAndSequence[float] | None = None,
        planes: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Metrics(self, distorted: VideoNode, mode: int | None = None) -> ConstantFormatVideoNode: ...
    def PackRGB(self) -> ConstantFormatVideoNode: ...
    def PlaneAverage(
        self,
        exclude: _SingleAndSequence[int],
        clipb: VideoNode | None = None,
        planes: _SingleAndSequence[int] | None = None,
        prop: _DataType | None = None,
    ) -> ConstantFormatVideoNode: ...
    def PlaneMinMax(
        self,
        minthr: float | None = None,
        maxthr: float | None = None,
        clipb: VideoNode | None = None,
        planes: _SingleAndSequence[int] | None = None,
        prop: _DataType | None = None,
    ) -> ConstantFormatVideoNode: ...
    def RFS(
        self,
        clipb: VideoNode,
        frames: _SingleAndSequence[int],
        mismatch: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...
    def SSIMULACRA2(self, distorted: VideoNode) -> ConstantFormatVideoNode: ...
    def XPSNR(
        self, distorted: VideoNode, temporal: int | None = None, verbose: int | None = None
    ) -> ConstantFormatVideoNode: ...

# end implementation

## implementation: warp

class _Plugin_warp_Core_Bound(Plugin):
    """This class implements the module definitions for the "warp" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def ABlur(
        self,
        clip: VideoNode,
        blur: int | None = None,
        type: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def ASobel(
        self,
        clip: VideoNode,
        thresh: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def AWarp(
        self,
        clip: VideoNode,
        mask: VideoNode,
        depth: _SingleAndSequence[int] | None = None,
        chroma: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
        opt: int | None = None,
        cplace: _DataType | None = None,
    ) -> ConstantFormatVideoNode: ...
    def AWarpSharp2(
        self,
        clip: VideoNode,
        thresh: int | None = None,
        blur: int | None = None,
        type: int | None = None,
        depth: _SingleAndSequence[int] | None = None,
        chroma: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
        opt: int | None = None,
        cplace: _DataType | None = None,
    ) -> ConstantFormatVideoNode: ...

class _Plugin_warp_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "warp" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def ABlur(
        self,
        blur: int | None = None,
        type: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
        opt: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def ASobel(
        self, thresh: int | None = None, planes: _SingleAndSequence[int] | None = None, opt: int | None = None
    ) -> ConstantFormatVideoNode: ...
    def AWarp(
        self,
        mask: VideoNode,
        depth: _SingleAndSequence[int] | None = None,
        chroma: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
        opt: int | None = None,
        cplace: _DataType | None = None,
    ) -> ConstantFormatVideoNode: ...
    def AWarpSharp2(
        self,
        thresh: int | None = None,
        blur: int | None = None,
        type: int | None = None,
        depth: _SingleAndSequence[int] | None = None,
        chroma: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
        opt: int | None = None,
        cplace: _DataType | None = None,
    ) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: warpsf

class _Plugin_warpsf_Core_Bound(Plugin):
    """This class implements the module definitions for the "warpsf" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def ABlur(
        self,
        clip: VideoNode,
        blur: int | None = None,
        type: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...
    def ASobel(
        self, clip: VideoNode, thresh: float | None = None, planes: _SingleAndSequence[int] | None = None
    ) -> ConstantFormatVideoNode: ...
    def AWarp(
        self,
        clip: VideoNode,
        mask: VideoNode,
        depth: _SingleAndSequence[int] | None = None,
        chroma: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...

class _Plugin_warpsf_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "warpsf" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def ABlur(
        self, blur: int | None = None, type: int | None = None, planes: _SingleAndSequence[int] | None = None
    ) -> ConstantFormatVideoNode: ...
    def ASobel(
        self, thresh: float | None = None, planes: _SingleAndSequence[int] | None = None
    ) -> ConstantFormatVideoNode: ...
    def AWarp(
        self,
        mask: VideoNode,
        depth: _SingleAndSequence[int] | None = None,
        chroma: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: wnnm

class _Plugin_wnnm_Core_Bound(Plugin):
    """This class implements the module definitions for the "wnnm" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def VAggregate(
        self, clip: VideoNode, src: VideoNode, planes: _SingleAndSequence[int], internal: int | None = None
    ) -> ConstantFormatVideoNode: ...
    def Version(self) -> dict[str, bytes]: ...
    def WNNM(
        self,
        clip: VideoNode,
        sigma: _SingleAndSequence[float] | None = None,
        block_size: int | None = None,
        block_step: int | None = None,
        group_size: int | None = None,
        bm_range: int | None = None,
        radius: int | None = None,
        ps_num: int | None = None,
        ps_range: int | None = None,
        residual: int | None = None,
        adaptive_aggregation: int | None = None,
        rclip: VideoNode | None = None,
    ) -> ConstantFormatVideoNode: ...
    def WNNMRaw(
        self,
        clip: VideoNode,
        sigma: _SingleAndSequence[float] | None = None,
        block_size: int | None = None,
        block_step: int | None = None,
        group_size: int | None = None,
        bm_range: int | None = None,
        radius: int | None = None,
        ps_num: int | None = None,
        ps_range: int | None = None,
        residual: int | None = None,
        adaptive_aggregation: int | None = None,
        rclip: VideoNode | None = None,
    ) -> ConstantFormatVideoNode: ...

class _Plugin_wnnm_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "wnnm" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def VAggregate(
        self, src: VideoNode, planes: _SingleAndSequence[int], internal: int | None = None
    ) -> ConstantFormatVideoNode: ...
    def WNNM(
        self,
        sigma: _SingleAndSequence[float] | None = None,
        block_size: int | None = None,
        block_step: int | None = None,
        group_size: int | None = None,
        bm_range: int | None = None,
        radius: int | None = None,
        ps_num: int | None = None,
        ps_range: int | None = None,
        residual: int | None = None,
        adaptive_aggregation: int | None = None,
        rclip: VideoNode | None = None,
    ) -> ConstantFormatVideoNode: ...
    def WNNMRaw(
        self,
        sigma: _SingleAndSequence[float] | None = None,
        block_size: int | None = None,
        block_step: int | None = None,
        group_size: int | None = None,
        bm_range: int | None = None,
        radius: int | None = None,
        ps_num: int | None = None,
        ps_range: int | None = None,
        residual: int | None = None,
        adaptive_aggregation: int | None = None,
        rclip: VideoNode | None = None,
    ) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: wwxd

class _Plugin_wwxd_Core_Bound(Plugin):
    """This class implements the module definitions for the "wwxd" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def WWXD(self, clip: VideoNode) -> ConstantFormatVideoNode: ...

class _Plugin_wwxd_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "wwxd" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def WWXD(self) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: znedi3

class _Plugin_znedi3_Core_Bound(Plugin):
    """This class implements the module definitions for the "znedi3" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def nnedi3(
        self,
        clip: VideoNode,
        field: int,
        dh: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
        nsize: int | None = None,
        nns: int | None = None,
        qual: int | None = None,
        etype: int | None = None,
        pscrn: int | None = None,
        opt: int | None = None,
        int16_prescreener: int | None = None,
        int16_predictor: int | None = None,
        exp: int | None = None,
        show_mask: int | None = None,
        x_nnedi3_weights_bin: _DataType | None = None,
        x_cpu: _DataType | None = None,
    ) -> ConstantFormatVideoNode: ...

class _Plugin_znedi3_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "znedi3" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def nnedi3(
        self,
        field: int,
        dh: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
        nsize: int | None = None,
        nns: int | None = None,
        qual: int | None = None,
        etype: int | None = None,
        pscrn: int | None = None,
        opt: int | None = None,
        int16_prescreener: int | None = None,
        int16_predictor: int | None = None,
        exp: int | None = None,
        show_mask: int | None = None,
        x_nnedi3_weights_bin: _DataType | None = None,
        x_cpu: _DataType | None = None,
    ) -> ConstantFormatVideoNode: ...

# end implementation

# implementation: zsmooth

class _Plugin_zsmooth_Core_Bound(Plugin):
    """This class implements the module definitions for the "zsmooth" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def BackwardClense(
        self, clip: VideoNode, planes: _SingleAndSequence[int] | None = None
    ) -> ConstantFormatVideoNode: ...
    def CCD(
        self,
        clip: VideoNode,
        threshold: float | None = None,
        temporal_radius: int | None = None,
        points: _SingleAndSequence[int] | None = None,
        scale: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Clense(
        self,
        clip: VideoNode,
        previous: VideoNode | None = None,
        next: VideoNode | None = None,
        planes: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...
    def DegrainMedian(
        self,
        clip: VideoNode,
        limit: _SingleAndSequence[float] | None = None,
        mode: _SingleAndSequence[int] | None = None,
        interlaced: int | None = None,
        norow: int | None = None,
        scalep: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def FluxSmoothST(
        self,
        clip: VideoNode,
        temporal_threshold: _SingleAndSequence[float] | None = None,
        spatial_threshold: _SingleAndSequence[float] | None = None,
        planes: _SingleAndSequence[int] | None = None,
        scalep: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def FluxSmoothT(
        self,
        clip: VideoNode,
        temporal_threshold: _SingleAndSequence[float] | None = None,
        planes: _SingleAndSequence[int] | None = None,
        scalep: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def ForwardClense(
        self, clip: VideoNode, planes: _SingleAndSequence[int] | None = None
    ) -> ConstantFormatVideoNode: ...
    def InterQuartileMean(
        self,
        clip: VideoNode,
        radius: _SingleAndSequence[int] | None = None,
        planes: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Median(
        self,
        clip: VideoNode,
        radius: _SingleAndSequence[int] | None = None,
        planes: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...
    def RemoveGrain(self, clip: VideoNode, mode: _SingleAndSequence[int]) -> ConstantFormatVideoNode: ...
    def Repair(
        self, clip: VideoNode, repairclip: VideoNode, mode: _SingleAndSequence[int]
    ) -> ConstantFormatVideoNode: ...
    def SmartMedian(
        self,
        clip: VideoNode,
        radius: _SingleAndSequence[int] | None = None,
        threshold: _SingleAndSequence[float] | None = None,
        scalep: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...
    def TemporalMedian(
        self, clip: VideoNode, radius: int | None = None, planes: _SingleAndSequence[int] | None = None
    ) -> ConstantFormatVideoNode: ...
    def TemporalRepair(
        self,
        clip: VideoNode,
        repairclip: VideoNode,
        mode: _SingleAndSequence[int] | None = None,
        planes: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...
    def TemporalSoften(
        self,
        clip: VideoNode,
        radius: int | None = None,
        threshold: _SingleAndSequence[float] | None = None,
        scenechange: int | None = None,
        scalep: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def TTempSmooth(
        self,
        clip: VideoNode,
        maxr: int | None = None,
        thresh: _SingleAndSequence[int] | None = None,
        mdiff: _SingleAndSequence[int] | None = None,
        strength: int | None = None,
        scthresh: float | None = None,
        fp: int | None = None,
        pfclip: VideoNode | None = None,
        planes: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...
    def VerticalCleaner(self, clip: VideoNode, mode: _SingleAndSequence[int]) -> ConstantFormatVideoNode: ...

class _Plugin_zsmooth_VideoNode_Bound(Plugin):
    """This class implements the module definitions for the "zsmooth" VapourSynth plugin.\n\n*This class cannot be imported.*"""
    def BackwardClense(self, planes: _SingleAndSequence[int] | None = None) -> ConstantFormatVideoNode: ...
    def CCD(
        self,
        threshold: float | None = None,
        temporal_radius: int | None = None,
        points: _SingleAndSequence[int] | None = None,
        scale: float | None = None,
    ) -> ConstantFormatVideoNode: ...
    def Clense(
        self,
        previous: VideoNode | None = None,
        next: VideoNode | None = None,
        planes: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...
    def DegrainMedian(
        self,
        limit: _SingleAndSequence[float] | None = None,
        mode: _SingleAndSequence[int] | None = None,
        interlaced: int | None = None,
        norow: int | None = None,
        scalep: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def FluxSmoothST(
        self,
        temporal_threshold: _SingleAndSequence[float] | None = None,
        spatial_threshold: _SingleAndSequence[float] | None = None,
        planes: _SingleAndSequence[int] | None = None,
        scalep: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def FluxSmoothT(
        self,
        temporal_threshold: _SingleAndSequence[float] | None = None,
        planes: _SingleAndSequence[int] | None = None,
        scalep: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def ForwardClense(self, planes: _SingleAndSequence[int] | None = None) -> ConstantFormatVideoNode: ...
    def InterQuartileMean(
        self, radius: _SingleAndSequence[int] | None = None, planes: _SingleAndSequence[int] | None = None
    ) -> ConstantFormatVideoNode: ...
    def Median(
        self, radius: _SingleAndSequence[int] | None = None, planes: _SingleAndSequence[int] | None = None
    ) -> ConstantFormatVideoNode: ...
    def RemoveGrain(self, mode: _SingleAndSequence[int]) -> ConstantFormatVideoNode: ...
    def Repair(self, repairclip: VideoNode, mode: _SingleAndSequence[int]) -> ConstantFormatVideoNode: ...
    def SmartMedian(
        self,
        radius: _SingleAndSequence[int] | None = None,
        threshold: _SingleAndSequence[float] | None = None,
        scalep: int | None = None,
        planes: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...
    def TemporalMedian(
        self, radius: int | None = None, planes: _SingleAndSequence[int] | None = None
    ) -> ConstantFormatVideoNode: ...
    def TemporalRepair(
        self,
        repairclip: VideoNode,
        mode: _SingleAndSequence[int] | None = None,
        planes: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...
    def TemporalSoften(
        self,
        radius: int | None = None,
        threshold: _SingleAndSequence[float] | None = None,
        scenechange: int | None = None,
        scalep: int | None = None,
    ) -> ConstantFormatVideoNode: ...
    def TTempSmooth(
        self,
        maxr: int | None = None,
        thresh: _SingleAndSequence[int] | None = None,
        mdiff: _SingleAndSequence[int] | None = None,
        strength: int | None = None,
        scthresh: float | None = None,
        fp: int | None = None,
        pfclip: VideoNode | None = None,
        planes: _SingleAndSequence[int] | None = None,
    ) -> ConstantFormatVideoNode: ...
    def VerticalCleaner(self, mode: _SingleAndSequence[int]) -> ConstantFormatVideoNode: ...

# end implementation

class RawNode:
    def get_frame(self, n: int) -> RawFrame: ...
    @overload
    def get_frame_async(self, n: int, cb: None = None) -> _Future[RawFrame]: ...
    @overload
    def get_frame_async(self, n: int, cb: Callable[[RawFrame | None, Exception | None], None]) -> None: ...
    def frames(
        self, prefetch: int | None = None, backlog: int | None = None, close: bool = False
    ) -> Iterator[RawFrame]: ...
    def clear_cache(self) -> None: ...
    def set_output(self, index: int = 0) -> None: ...
    def is_inspectable(self, version: int | None = None) -> bool: ...

    if not TYPE_CHECKING:
        @property
        def _node_name(self) -> str: ...
        @property
        def _name(self) -> str: ...
        @property
        def _inputs(self) -> dict[str, _VapourSynthMapValue]: ...
        @property
        def _timings(self) -> int: ...
        @property
        def _mode(self) -> FilterMode: ...
        @property
        def _dependencies(self): ...

    @overload
    def __eq__(self: Self, other: Self, /) -> bool: ...
    @overload
    def __eq__(self, other: Any, /) -> Literal[False]: ...
    def __hash__(self) -> int: ...
    def __add__(self, other: Self, /) -> Self: ...
    def __radd__(self, other: Self, /) -> Self: ...
    def __mul__(self, other: int) -> Self: ...
    def __rmul__(self, other: int) -> Self: ...
    def __getitem__(self, index: int | slice, /) -> Self: ...
    def __len__(self) -> int: ...

class VideoNode(RawNode):
    format: VideoFormat | None

    width: int
    height: int

    fps_num: int
    fps_den: int

    fps: Fraction

    num_frames: int

    def get_frame(self, n: int) -> VideoFrame: ...
    @overload  # type: ignore[override]
    def get_frame_async(self, n: int, cb: None = None) -> _Future[VideoFrame]: ...
    @overload
    def get_frame_async(self, n: int, cb: Callable[[VideoFrame | None, Exception | None], None]) -> None: ...
    def frames(
        self, prefetch: int | None = None, backlog: int | None = None, close: bool = False
    ) -> Iterator[VideoFrame]: ...
    def set_output(self, index: int = 0, alpha: Self | None = None, alt_output: Literal[0, 1, 2] = 0) -> None: ...
    def output(
        self,
        fileobj: BinaryIO,
        y4m: bool = False,
        progress_update: Callable[[int, int], None] | None = None,
        prefetch: int = 0,
        backlog: int = -1,
    ) -> None: ...

    # instance_bound_VideoNode: adg
    @property
    def adg(self) -> _Plugin_adg_VideoNode_Bound:
        """Adaptive grain"""
    # end instance
    # instance_bound_VideoNode: akarin
    @property
    def akarin(self) -> _Plugin_akarin_VideoNode_Bound:
        """Akarin's Experimental Filters"""
    # end instance
    # instance_bound_VideoNode: bilateralgpu
    @property
    def bilateralgpu(self) -> _Plugin_bilateralgpu_VideoNode_Bound:
        """Bilateral filter using CUDA"""
    # end instance
    # instance_bound_VideoNode: bilateralgpu_rtc
    @property
    def bilateralgpu_rtc(self) -> _Plugin_bilateralgpu_rtc_VideoNode_Bound:
        """Bilateral filter using CUDA (NVRTC)"""
    # end instance
    # instance_bound_VideoNode: bm3d
    @property
    def bm3d(self) -> _Plugin_bm3d_VideoNode_Bound:
        """Implementation of BM3D denoising filter for VapourSynth."""
    # end instance
    # instance_bound_VideoNode: bm3dcpu
    @property
    def bm3dcpu(self) -> _Plugin_bm3dcpu_VideoNode_Bound:
        """BM3D algorithm implemented in AVX and AVX2 intrinsics"""
    # end instance
    # instance_bound_VideoNode: bm3dcuda
    @property
    def bm3dcuda(self) -> _Plugin_bm3dcuda_VideoNode_Bound:
        """BM3D algorithm implemented in CUDA"""
    # end instance
    # instance_bound_VideoNode: bm3dcuda_rtc
    @property
    def bm3dcuda_rtc(self) -> _Plugin_bm3dcuda_rtc_VideoNode_Bound:
        """BM3D algorithm implemented in CUDA (NVRTC)"""
    # end instance
    # instance_bound_VideoNode: bm3dhip
    @property
    def bm3dhip(self) -> _Plugin_bm3dhip_VideoNode_Bound:
        """BM3D algorithm implemented in HIP (AMD)"""
    # end instance
    # instance_bound_VideoNode: bm3dhip
    @property
    def bm3dsycl(self) -> _Plugin_bm3dsycl_VideoNode_Bound:
        """BM3D algorithm implemented in SYCL"""
    # end instance
    # instance_bound_VideoNode: bwdif
    @property
    def bwdif(self) -> _Plugin_bwdif_VideoNode_Bound:
        """BobWeaver Deinterlacing Filter"""
    # end instance
    # instance_bound_VideoNode: cs
    @property
    def cs(self) -> _Plugin_cs_VideoNode_Bound:
        """carefulsource"""
    # end instance
    # instance_bound_VideoNode: dctf
    @property
    def dctf(self) -> _Plugin_dctf_VideoNode_Bound:
        """DCT/IDCT Frequency Suppressor"""
    # end instance
    # instance_bound_VideoNode: deblock
    @property
    def deblock(self) -> _Plugin_deblock_VideoNode_Bound:
        """It does a deblocking of the picture, using the deblocking filter of h264"""
    # end instance
    # instance_bound_VideoNode: descale
    @property
    def descale(self) -> _Plugin_descale_VideoNode_Bound:
        """Undo linear interpolation"""
    # end instance
    # instance_bound_VideoNode: dfttest
    @property
    def dfttest(self) -> _Plugin_dfttest_VideoNode_Bound:
        """2D/3D frequency domain denoiser"""
    # end instance
    # instance_bound_VideoNode: dfttest2_cpu
    @property
    def dfttest2_cpu(self) -> _Plugin_dfttest2_cpu_VideoNode_Bound:
        """DFTTest2 (CPU)"""
    # end instance
    # instance_bound_VideoNode: dfttest2_cuda
    @property
    def dfttest2_cuda(self) -> _Plugin_dfttest2_cuda_VideoNode_Bound:
        """DFTTest2 (CUDA)"""
    # end instance
    # instance_bound_VideoNode: dfttest2_nvrtc
    @property
    def dfttest2_nvrtc(self) -> _Plugin_dfttest2_nvrtc_VideoNode_Bound:
        """DFTTest2 (NVRTC)"""
    # end instance
    # instance_bound_VideoNode: edgemasks
    @property
    def edgemasks(self) -> _Plugin_edgemasks_VideoNode_Bound:
        """Creates an edge mask using various operators"""
    # end instance
    # instance_bound_VideoNode: eedi2
    @property
    def eedi2(self) -> _Plugin_eedi2_VideoNode_Bound:
        """EEDI2"""
    # end instance
    # instance_bound_VideoNode: eedi2cuda
    @property
    def eedi2cuda(self) -> _Plugin_eedi2cuda_VideoNode_Bound:
        """EEDI2 filter using CUDA"""
    # end instance
    # instance_bound_VideoNode: eedi3m
    @property
    def eedi3m(self) -> _Plugin_eedi3m_VideoNode_Bound:
        """Enhanced Edge Directed Interpolation 3"""
    # end instance
    # instance_bound_VideoNode: fft3dfilter
    @property
    def fft3dfilter(self) -> _Plugin_fft3dfilter_VideoNode_Bound:
        """systems"""
    # end instance
    # instance_bound_VideoNode: fmtc
    @property
    def fmtc(self) -> _Plugin_fmtc_VideoNode_Bound:
        """Format converter"""
    # end instance
    # instance_bound_VideoNode: hysteresis
    @property
    def hysteresis(self) -> _Plugin_hysteresis_VideoNode_Bound:
        """Hysteresis filter."""
    # end instance
    # instance_bound_VideoNode: imwri
    @property
    def imwri(self) -> _Plugin_imwri_VideoNode_Bound:
        """VapourSynth ImageMagick 7 HDRI Writer/Reader"""
    # end instance
    # instance_bound_VideoNode: knlm
    @property
    def knlm(self) -> _Plugin_knlm_VideoNode_Bound:
        """KNLMeansCL for VapourSynth"""
    # end instance
    # instance_bound_VideoNode: manipmv
    @property
    def manipmv(self) -> _Plugin_manipmv_VideoNode_Bound:
        """Manipulate Motion Vectors"""
    # end instance
    # instance_bound_VideoNode: mv
    @property
    def mv(self) -> _Plugin_mv_VideoNode_Bound:
        """MVTools v24"""
    # end instance
    # instance_bound_VideoNode: mvsf
    @property
    def mvsf(self) -> _Plugin_mvsf_VideoNode_Bound:
        """MVTools Single Precision"""
    # end instance
    # instance_bound_VideoNode: neo_f3kdb
    @property
    def neo_f3kdb(self) -> _Plugin_neo_f3kdb_VideoNode_Bound:
        """Neo F3KDB Deband Filter r9"""
    # end instance
    # instance_bound_VideoNode: nlm_cuda
    @property
    def nlm_cuda(self) -> _Plugin_nlm_cuda_VideoNode_Bound:
        """Non-local means denoise filter implemented in CUDA"""
    # end instance
    # instance_bound_VideoNode: nlm_ispc
    @property
    def nlm_ispc(self) -> _Plugin_nlm_ispc_VideoNode_Bound:
        """Non-local means denoise filter implemented in ISPC"""
    # end instance
    # instance_bound_VideoNode: noise
    @property
    def noise(self) -> _Plugin_noise_VideoNode_Bound:
        """Noise generator"""
    # end instance
    # instance_bound_VideoNode: placebo
    @property
    def placebo(self) -> _Plugin_placebo_VideoNode_Bound:
        """libplacebo plugin for VapourSynth"""
    # end instance
    # instance_bound_VideoNode: resize
    @property
    def resize(self) -> _Plugin_resize_VideoNode_Bound:
        """VapourSynth Resize"""
    # end instance
    # instance_bound_VideoNode: resize2
    @property
    def resize2(self) -> _Plugin_resize2_VideoNode_Bound:
        """Built-in VapourSynth resizer based on zimg with some modifications."""
    # end instance
    # instance_bound_VideoNode: sangnom
    @property
    def sangnom(self) -> _Plugin_sangnom_VideoNode_Bound:
        """VapourSynth Single Field Deinterlacer"""
    # end instance
    # instance_bound_VideoNode: scxvid
    @property
    def scxvid(self) -> _Plugin_scxvid_VideoNode_Bound:
        """VapourSynth Scxvid Plugin"""
    # end instance
    @property
    def sneedif(self) -> _Plugin_sneedif_VideoNode_Bound:
        """Setsugen No Ensemble of Edge Directed Interpolation Functions"""
    # end instance
    # instance_bound_VideoNode: std
    @property
    def std(self) -> _Plugin_std_VideoNode_Bound[VideoNode]:
        """VapourSynth Core Functions"""
    # end instance
    # instance_bound_VideoNode: sub
    @property
    def sub(self) -> _Plugin_sub_VideoNode_Bound:
        """A subtitling filter based on libass and FFmpeg."""
    # end instance
    # instance_bound_VideoNode: tcanny
    @property
    def tcanny(self) -> _Plugin_tcanny_VideoNode_Bound:
        """Build an edge map using canny edge detection"""
    # end instance
    # instance_bound_VideoNode: tedgemask
    @property
    def tedgemask(self) -> _Plugin_tedgemask_VideoNode_Bound:
        """Edge detection plugin"""
    # end instance
    # instance_bound_VideoNode: text
    @property
    def text(self) -> _Plugin_text_VideoNode_Bound:
        """VapourSynth Text"""
    # end instance
    # instance_bound_VideoNode: vivtc
    @property
    def vivtc(self) -> _Plugin_vivtc_VideoNode_Bound:
        """VFM"""
    # end instance
    # instance_bound_VideoNode: vszip
    @property
    def vszip(self) -> _Plugin_vszip_VideoNode_Bound:
        """VapourSynth Zig Image Process"""
    # end instance
    # instance_bound_VideoNode: warp
    @property
    def warp(self) -> _Plugin_warp_VideoNode_Bound:
        """Sharpen images by warping"""
    # end instance
    # instance_bound_VideoNode: warpsf
    @property
    def warpsf(self) -> _Plugin_warpsf_VideoNode_Bound:
        """Warpsharp floating point version"""
    # end instance
    # instance_bound_VideoNode: wnnm
    @property
    def wnnm(self) -> _Plugin_wnnm_VideoNode_Bound:
        """Weighted Nuclear Norm Minimization Denoiser"""
    # end instance
    # instance_bound_VideoNode: wwxd
    @property
    def wwxd(self) -> _Plugin_wwxd_VideoNode_Bound:
        """Scene change detection approximately like Xvid's"""
    # end instance
    # instance_bound_VideoNode: znedi3
    @property
    def znedi3(self) -> _Plugin_znedi3_VideoNode_Bound:
        """Neural network edge directed interpolation (3rd gen.)"""
    # end instance
    # instance_bound_VideoNode: zsmooth
    @property
    def zsmooth(self) -> _Plugin_zsmooth_VideoNode_Bound:
        """Smoothing functions in Zig"""
    # end instance

_VideoNodeT = TypeVar("_VideoNodeT", bound=VideoNode)

class AudioNode(RawNode):
    sample_type: SampleType
    bits_per_sample: int
    bytes_per_sample: int

    channel_layout: int
    num_channels: int

    sample_rate: int
    num_samples: int

    num_frames: int

    def get_frame(self, n: int) -> AudioFrame: ...
    @overload  # type: ignore[override]
    def get_frame_async(self, n: int, cb: None = None) -> _Future[AudioFrame]: ...
    @overload
    def get_frame_async(self, n: int, cb: Callable[[AudioFrame | None, Exception | None], None]) -> None: ...
    def frames(
        self, prefetch: int | None = None, backlog: int | None = None, close: bool = False
    ) -> Iterator[AudioFrame]: ...
    @property
    def channels(self) -> ChannelLayout: ...

    # instance_bound_AudioNode: std
    @property
    def std(self) -> _Plugin_std_AudioNode_Bound:
        """VapourSynth Core Functions"""
    # end instance

class Core:
    @property
    def num_threads(self) -> int: ...
    @num_threads.setter
    def num_threads(self, value: int) -> None: ...
    @property
    def max_cache_size(self) -> int: ...
    @max_cache_size.setter
    def max_cache_size(self, mb: int) -> None: ...
    @property
    def flags(self) -> int: ...
    def plugins(self) -> Iterator[Plugin]: ...
    def query_video_format(
        self,
        color_family: ColorFamily,
        sample_type: SampleType,
        bits_per_sample: int,
        subsampling_w: int = 0,
        subsampling_h: int = 0,
    ) -> VideoFormat: ...
    def get_video_format(self, id: VideoFormat | int | PresetVideoFormat) -> VideoFormat: ...
    def create_video_frame(self, format: VideoFormat, width: int, height: int) -> VideoFrame: ...
    def log_message(self, message_type: MessageType, message: str) -> None: ...
    def add_log_handler(self, handler_func: Callable[[MessageType, str], None]) -> LogHandle: ...
    def remove_log_handler(self, handle: LogHandle) -> None: ...
    def clear_cache(self) -> None: ...
    def version(self) -> str: ...
    def version_number(self) -> int: ...

    # instance_bound_Core: adg
    @property
    def adg(self) -> _Plugin_adg_Core_Bound:
        """Adaptive grain"""
    # end instance
    # instance_bound_Core: akarin
    @property
    def akarin(self) -> _Plugin_akarin_Core_Bound:
        """Akarin's Experimental Filters"""
    # end instance
    # instance_bound_Core: bilateralgpu
    @property
    def bilateralgpu(self) -> _Plugin_bilateralgpu_Core_Bound:
        """Bilateral filter using CUDA"""
    # end instance
    # instance_bound_Core: bilateralgpu_rtc
    @property
    def bilateralgpu_rtc(self) -> _Plugin_bilateralgpu_rtc_Core_Bound:
        """Bilateral filter using CUDA (NVRTC)"""
    # end instance
    # instance_bound_Core: bm3d
    @property
    def bm3d(self) -> _Plugin_bm3d_Core_Bound:
        """Implementation of BM3D denoising filter for VapourSynth."""
    # end instance
    # instance_bound_Core: bm3dcpu
    @property
    def bm3dcpu(self) -> _Plugin_bm3dcpu_Core_Bound:
        """BM3D algorithm implemented in AVX and AVX2 intrinsics"""
    # end instance
    # instance_bound_Core: bm3dcuda
    @property
    def bm3dcuda(self) -> _Plugin_bm3dcuda_Core_Bound:
        """BM3D algorithm implemented in CUDA"""
    # end instance
    # instance_bound_Core: bm3dcuda_rtc
    @property
    def bm3dcuda_rtc(self) -> _Plugin_bm3dcuda_rtc_Core_Bound:
        """BM3D algorithm implemented in CUDA (NVRTC)"""
    # end instance
    # instance_bound_Core: bm3dhip
    @property
    def bm3dhip(self) -> _Plugin_bm3dhip_Core_Bound:
        """BM3D algorithm implemented in HIP (AMD)"""
    # end instance
    # instance_bound_Core: bm3dsycl
    @property
    def bm3dsycl(self) -> _Plugin_bm3dsycl_Core_Bound:
        """BM3D algorithm implemented in SYCL"""
    # end instance
    # instance_bound_Core: bs
    @property
    def bs(self) -> _Plugin_bs_Core_Bound:
        """Best Source 2"""
    # end instance
    # instance_bound_Core: bwdif
    @property
    def bwdif(self) -> _Plugin_bwdif_Core_Bound:
        """BobWeaver Deinterlacing Filter"""
    # end instance
    # instance_bound_Core: cs
    @property
    def cs(self) -> _Plugin_cs_Core_Bound:
        """carefulsource"""
    # end instance
    # instance_bound_Core: d2v
    @property
    def d2v(self) -> _Plugin_d2v_Core_Bound:
        """D2V Source"""
    # end instance
    # instance_bound_Core: dctf
    @property
    def dctf(self) -> _Plugin_dctf_Core_Bound:
        """DCT/IDCT Frequency Suppressor"""
    # end instance
    # instance_bound_Core: deblock
    @property
    def deblock(self) -> _Plugin_deblock_Core_Bound:
        """It does a deblocking of the picture, using the deblocking filter of h264"""
    # end instance
    # instance_bound_Core: descale
    @property
    def descale(self) -> _Plugin_descale_Core_Bound:
        """Undo linear interpolation"""
    # end instance
    # instance_bound_Core: dfttest
    @property
    def dfttest(self) -> _Plugin_dfttest_Core_Bound:
        """2D/3D frequency domain denoiser"""
    # end instance
    # instance_bound_Core: dfttest2_cpu
    @property
    def dfttest2_cpu(self) -> _Plugin_dfttest2_cpu_Core_Bound:
        """DFTTest2 (CPU)"""
    # end instance
    # instance_bound_Core: dfttest2_cuda
    @property
    def dfttest2_cuda(self) -> _Plugin_dfttest2_cuda_Core_Bound:
        """DFTTest2 (CUDA)"""
    # end instance
    # instance_bound_Core: dfttest2_nvrtc
    @property
    def dfttest2_nvrtc(self) -> _Plugin_dfttest2_nvrtc_Core_Bound:
        """DFTTest2 (NVRTC)"""
    # end instance
    # instance_bound_Core: dgdecodenv
    @property
    def dgdecodenv(self) -> _Plugin_dgdecodenv_Core_Bound:
        """DGDecodeNV for VapourSynth"""
    # end instance
    # instance_bound_Core: dvdsrc2
    @property
    def dvdsrc2(self) -> _Plugin_dvdsrc2_Core_Bound:
        """Dvdsrc 2nd tour"""
    # end instance
    # instance_bound_Core: edgemasks
    @property
    def edgemasks(self) -> _Plugin_edgemasks_Core_Bound:
        """Creates an edge mask using various operators"""
    # end instance
    # instance_bound_Core: eedi2
    @property
    def eedi2(self) -> _Plugin_eedi2_Core_Bound:
        """EEDI2"""
    # end instance
    # instance_bound_Core: eedi2cuda
    @property
    def eedi2cuda(self) -> _Plugin_eedi2cuda_Core_Bound:
        """EEDI2 filter using CUDA"""
    # end instance
    # instance_bound_Core: eedi3m
    @property
    def eedi3m(self) -> _Plugin_eedi3m_Core_Bound:
        """Enhanced Edge Directed Interpolation 3"""
    # end instance
    # instance_bound_Core: fft3dfilter
    @property
    def fft3dfilter(self) -> _Plugin_fft3dfilter_Core_Bound:
        """systems"""
    # end instance
    # instance_bound_Core: ffms2
    @property
    def ffms2(self) -> _Plugin_ffms2_Core_Bound:
        """FFmpegSource 2 for VapourSynth"""
    # end instance
    # instance_bound_Core: fmtc
    @property
    def fmtc(self) -> _Plugin_fmtc_Core_Bound:
        """Format converter"""
    # end instance
    # instance_bound_Core: hysteresis
    @property
    def hysteresis(self) -> _Plugin_hysteresis_Core_Bound:
        """Hysteresis filter."""
    # end instance
    # instance_bound_Core: imwri
    @property
    def imwri(self) -> _Plugin_imwri_Core_Bound:
        """VapourSynth ImageMagick 7 HDRI Writer/Reader"""
    # end instance
    # instance_bound_Core: knlm
    @property
    def knlm(self) -> _Plugin_knlm_Core_Bound:
        """KNLMeansCL for VapourSynth"""
    # end instance
    # instance_bound_Core: lsmas
    @property
    def lsmas(self) -> _Plugin_lsmas_Core_Bound:
        """LSMASHSource for VapourSynth"""
    # end instance
    # instance_bound_Core: manipmv
    @property
    def manipmv(self) -> _Plugin_manipmv_Core_Bound:
        """Manipulate Motion Vectors"""
    # end instance
    # instance_bound_Core: mv
    @property
    def mv(self) -> _Plugin_mv_Core_Bound:
        """MVTools v24"""
    # end instance
    # instance_bound_Core: mvsf
    @property
    def mvsf(self) -> _Plugin_mvsf_Core_Bound:
        """MVTools Single Precision"""
    # end instance
    # instance_bound_Core: neo_f3kdb
    @property
    def neo_f3kdb(self) -> _Plugin_neo_f3kdb_Core_Bound:
        """Neo F3KDB Deband Filter r9"""
    # end instance
    # instance_bound_Core: nlm_cuda
    @property
    def nlm_cuda(self) -> _Plugin_nlm_cuda_Core_Bound:
        """Non-local means denoise filter implemented in CUDA"""
    # end instance
    # instance_bound_Core: nlm_ispc
    @property
    def nlm_ispc(self) -> _Plugin_nlm_ispc_Core_Bound:
        """Non-local means denoise filter implemented in ISPC"""
    # end instance
    @property
    def noise(self) -> _Plugin_noise_Core_Bound:
        """Noise generator"""
    # end instance
    # instance_bound_Core: placebo
    @property
    def placebo(self) -> _Plugin_placebo_Core_Bound:
        """libplacebo plugin for VapourSynth"""
    # end instance
    # instance_bound_Core: resize
    @property
    def resize(self) -> _Plugin_resize_Core_Bound:
        """VapourSynth Resize"""
    # end instance
    # instance_bound_Core: resize2
    @property
    def resize2(self) -> _Plugin_resize2_Core_Bound:
        """Built-in VapourSynth resizer based on zimg with some modifications."""
    # end instance
    # instance_bound_Core: sangnom
    @property
    def sangnom(self) -> _Plugin_sangnom_Core_Bound:
        """VapourSynth Single Field Deinterlacer"""
    # end instance
    # instance_bound_Core: scxvid
    @property
    def scxvid(self) -> _Plugin_scxvid_Core_Bound:
        """VapourSynth Scxvid Plugin"""
    # end instance
    # instance_bound_Core: sneedif
    @property
    def sneedif(self) -> _Plugin_sneedif_Core_Bound:
        """Setsugen No Ensemble of Edge Directed Interpolation Functions"""
    # end instance
    # instance_bound_Core: std
    @property
    def std(self) -> _Plugin_std_Core_Bound:
        """VapourSynth Core Functions"""
    # end instance
    # instance_bound_Core: sub
    @property
    def sub(self) -> _Plugin_sub_Core_Bound:
        """A subtitling filter based on libass and FFmpeg."""
    # end instance
    @property
    def tcanny(self) -> _Plugin_tcanny_Core_Bound:
        """Build an edge map using canny edge detection"""
    # end instance
    # instance_bound_Core: tedgemask
    @property
    def tedgemask(self) -> _Plugin_tedgemask_Core_Bound:
        """Edge detection plugin"""
    # end instance
    # instance_bound_Core: text
    @property
    def text(self) -> _Plugin_text_Core_Bound:
        """VapourSynth Text"""
    # end instance
    # instance_bound_Core: vivtc
    @property
    def vivtc(self) -> _Plugin_vivtc_Core_Bound:
        """VFM"""
    # end instance
    # instance_bound_Core: vszip
    @property
    def vszip(self) -> _Plugin_vszip_Core_Bound:
        """VapourSynth Zig Image Process"""
    # end instance
    # instance_bound_Core: warp
    @property
    def warp(self) -> _Plugin_warp_Core_Bound:
        """Sharpen images by warping"""
    # end instance
    # instance_bound_Core: warpsf
    @property
    def warpsf(self) -> _Plugin_warpsf_Core_Bound:
        """Warpsharp floating point version"""
    # end instance
    # instance_bound_Core: wnnm
    @property
    def wnnm(self) -> _Plugin_wnnm_Core_Bound:
        """Weighted Nuclear Norm Minimization Denoiser"""
    # end instance
    # instance_bound_Core: wwxd
    @property
    def wwxd(self) -> _Plugin_wwxd_Core_Bound:
        """Scene change detection approximately like Xvid's"""
    # end instance
    # instance_bound_Core: znedi3
    @property
    def znedi3(self) -> _Plugin_znedi3_Core_Bound:
        """Neural network edge directed interpolation (3rd gen.)"""
    # end instance
    # instance_bound_Core: zsmooth
    @property
    def zsmooth(self) -> _Plugin_zsmooth_Core_Bound:
        """Smoothing functions in Zig"""
    # end instance

class _CoreProxy(Core):
    @property
    def core(self) -> Core: ...

core: _CoreProxy
