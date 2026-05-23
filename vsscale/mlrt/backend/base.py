from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

type Shape = tuple[int, int]


@dataclass(kw_only=True, frozen=True)
class Backend:
    supports_onnx_serialization: ClassVar[bool] = True
