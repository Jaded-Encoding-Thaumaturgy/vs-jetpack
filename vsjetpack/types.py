import sys

__all__ = ["TypeIs", "TypeVar", "deprecated"]

if sys.version_info < (3, 13):
    import typing_extensions

    TypeIs = typing_extensions.TypeIs
    TypeVar = typing_extensions.TypeVar
    deprecated = typing_extensions.deprecated
else:
    from typing import TypeIs, TypeVar
    from warnings import deprecated
