from __future__ import annotations

from vstools import CustomIntEnum

__all__ = ["GuidedFilterMode"]


class GuidedFilterMode(CustomIntEnum):
    ORIGINAL = 0
    """Original Guided Filter"""

    WEIGHTED = 1
    """Weighted Guided Image Filter"""

    GRADIENT = 2
    """Gradient Domain Guided Image Filter"""
