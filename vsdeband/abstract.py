from __future__ import annotations

from abc import abstractmethod
from typing import Any

from vstools import inject_self, vs

__all__ = ["Debander"]


class Debander:
    @abstractmethod
    @inject_self
    def deband(self, clip: vs.VideoNode, **kwargs: Any) -> vs.VideoNode: ...
