from __future__ import annotations

from typing import TYPE_CHECKING

from . import base, migx, ncnn, ort, ov, trt

if TYPE_CHECKING:

    class Backend(base.Backend):
        class OV(base.Backend):
            class CPU(ov.CPU): ...

            class GPU(ov.GPU): ...

            class NPU(ov.NPU): ...

        class ORT(base.Backend):
            class CPU(ort.CPU): ...

            class CUDA(ort.CUDA): ...

            class DML(ort.DML): ...

            class CoreML(ort.CoreML): ...

        class NCNN(base.Backend):
            class VK(ncnn.VK): ...

        class TRT(trt.TRT):
            class RTX(trt.RTX): ...

        class MIGX(migx.MIGX): ...
else:
    Backend = base.Backend

    Backend.OV = ov.OV
    Backend.OV.CPU = ov.CPU
    Backend.OV.GPU = ov.GPU
    Backend.OV.NPU = ov.NPU

    Backend.ORT = ort.ORT
    Backend.ORT.CPU = ort.CPU
    Backend.ORT.CUDA = ort.CUDA
    Backend.ORT.DML = ort.DML
    Backend.ORT.CoreML = ort.CoreML

    Backend.NCNN = ncnn.NCNN
    Backend.NCNN.VK = ncnn.VK

    Backend.TRT = trt.TRT
    Backend.TRT.RTX = trt.RTX

    Backend.MIGX = migx.MIGX

__all__ = ["Backend"]
