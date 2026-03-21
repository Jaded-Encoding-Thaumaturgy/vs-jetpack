import sys

__all__ = ["TypeIs", "TypeVar", "deprecated"]

if sys.version_info >= (3, 13):
    from typing import TypeIs, TypeVar
    from warnings import deprecated
else:
    import typing_extensions

    TypeIs = typing_extensions.TypeIs
    TypeVar = typing_extensions.TypeVar
    deprecated = typing_extensions.deprecated
