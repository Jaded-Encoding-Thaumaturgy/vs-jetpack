import sys
from typing import TypeIs, TypeVar
from warnings import deprecated

__all__ = ["TypeForm", "TypeIs", "TypeVar", "deprecated"]

if sys.version_info < (3, 15):
    from typing_extensions import TypeForm
else:
    from typing import TypeForm
