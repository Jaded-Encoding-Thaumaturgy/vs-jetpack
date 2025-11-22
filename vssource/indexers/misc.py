from __future__ import annotations

from typing import Any

from jetpytools import CustomIntEnum, SPathLike
from vapoursynth import VideoNode

from vstools import core

from .base import CacheIndexer, Indexer

__all__ = ["FFMS2", "IMWRI", "LSMAS", "BestSource", "CarefulSource", "ZipSource"]


# Video indexers
class BestSource(CacheIndexer):
    """
    [BestSource](https://github.com/vapoursynth/bestsource) indexer.

    Unlike the plugin's default behavior, the indexer cache file will be stored in `.vsjet/vssource`
    next to the script file.

    When ``cachemode`` is 0, 1, or 2 (NEVER, CACHE_PATH, or CACHE_PATH_WRITE), the behavior falls back
    to the default cache handling defined by the BestSource plugin itself.
    """

    _source_func = core.lazy.bs.VideoSource
    _cache_arg_name = "cachepath"
    _ext = None

    class CacheMode(CustomIntEnum):
        """
        Cache mode.
        """

        NEVER = 0
        """
        Never read or write index to disk.
        """

        CACHE_PATH = 1
        """
        Always try to read index but only write index to disk when it will make a noticeable difference
        on subsequent runs and store index files in a subtree of *cachepath*.
        """

        CACHE_PATH_WRITE = 2
        """
        Always try to read and write index to disk and store index files in a subtree of *cachepath*.
        """

        ABSOLUTE = 3
        """
        Always try to read index but only write index to disk when it will make a noticeable difference
        on subsequent runs and store index files in the absolute path in *cachepath*
        with track number and index extension appended.
        """

        ABSOLUTE_WRITE = 4
        """
        Always try to read and write index to disk and store index files
        in the absolute path in *cachepath* with track number and index extension appended.
        """

    def __init__(self, *, force: bool = True, cachemode: int = CacheMode.ABSOLUTE, **kwargs: Any) -> None:
        super().__init__(force=force, cachemode=cachemode, **kwargs)

    @classmethod
    def source_func(cls, path: SPathLike, **kwargs: Any) -> VideoNode:
        if kwargs["cachemode"] <= cls.CacheMode.CACHE_PATH_WRITE and cls._cache_arg_name not in kwargs:
            kwargs[cls._cache_arg_name] = None

        return super().source_func(path, **kwargs)


class FFMS2(CacheIndexer):
    """
    [FFmpegSource2](https://github.com/FFMS/ffms2) indexer.

    Unlike the plugin's default behavior, the indexer cache file will be stored in `.vsjet/vssource`
    next to the script file.
    """

    _source_func = core.lazy.ffms2.Source
    _cache_arg_name = "cachefile"
    _ext = ".ffindex"


class LSMAS(CacheIndexer):
    """
    [L-SMASH-Works](https://github.com/HomeOfAviSynthPlusEvolution/L-SMASH-Works) indexer.

    Unlike the plugin's default behavior, the indexer cache file will be stored in `.vsjet/vssource`
    next to the script file.
    """

    _source_func = core.lazy.lsmas.LWLibavSource
    _cache_arg_name = "cachefile"
    _ext = ".lwi"


class CarefulSource(Indexer):
    """
    CarefulSource indexer
    """

    _source_func = core.lazy.cs.ImageSource


# Image indexers
class IMWRI(Indexer):
    """
    ImageMagick Writer-Reader indexer
    """

    _source_func = core.lazy.imwri.Read


class ZipSource(Indexer):
    """
    vszip image reader indexer
    """

    _source_func = core.lazy.vszip.ImageRead
