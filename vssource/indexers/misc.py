from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from logging import INFO, Handler, Logger, LogRecord, getLogger
from typing import Any

from jetpytools import CustomIntEnum, SPathLike

from vsjetpack import require_jet_dependency
from vstools import core, vs

from .base import CacheIndexer, Indexer

__all__ = ["FFMS2", "IMWRI", "LSMAS", "BestSource", "CarefulSource", "ZipSource"]


# Video indexers
class BestSource(CacheIndexer):
    """
    [BestSource](https://github.com/vapoursynth/bestsource) indexer.

    Unlike the plugin's default behavior, the indexer cache file will be stored in `.vsjet/vssource`
    next to the script file.

    When `cachemode` is 0, 1, or 2 (NEVER, CACHE_PATH, or CACHE_PATH_WRITE) or `cachepath=None`,
    the behavior falls back to the default cache handling defined by the BestSource plugin itself.
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

    def __init__(
        self,
        *,
        cachemode: int = CacheMode.ABSOLUTE,
        rff: int | None = True,
        showprogress: int | None = True,
        show_pretty_progress: int | None = False,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            cachemode: The cache mode. See [here][vssource.BestSource] and [here][vssource.BestSource.CacheMode]
                for more explanation.
            rff: Apply RFF flags to the video. If the video doesn't have or use RFF flags, the output is unchanged.
            showprogress: Print indexing progress as VapourSynth information level log messages.
            show_pretty_progress: Display a rich-based progress bar if `showprogress` is also set to True.
        """
        super().__init__(
            cachemode=cachemode,
            rff=rff,
            showprogress=showprogress,
            show_pretty_progress=show_pretty_progress,
            **kwargs,
        )

    @classmethod
    def source_func(cls, path: SPathLike, **kwargs: Any) -> vs.VideoNode:
        if kwargs["cachemode"] <= cls.CacheMode.CACHE_PATH_WRITE and cls._cache_arg_name not in kwargs:
            kwargs[cls._cache_arg_name] = None

        if kwargs.pop("show_pretty_progress"):
            with _bs_pretty_progress():
                return super().source_func(path, **kwargs)

        return super().source_func(path, **kwargs)


class FFMS2(CacheIndexer):
    """
    [FFmpegSource2](https://github.com/FFMS/ffms2) indexer.

    Unlike the plugin's default behavior, the indexer cache file will be stored in `.vsjet/vssource`
    next to the script file.

    When `cachefile=None`, the behavior falls back to the default cache handling defined by the plugin itself.
    """

    _source_func = core.lazy.ffms2.Source
    _cache_arg_name = "cachefile"
    _ext = ".ffindex"


class LSMAS(CacheIndexer):
    """
    [L-SMASH-Works](https://github.com/HomeOfAviSynthPlusEvolution/L-SMASH-Works) indexer.

    Unlike the plugin's default behavior, the indexer cache file will be stored in `.vsjet/vssource`
    next to the script file.

    When `cachefile=None`, the behavior falls back to the default cache handling defined by the plugin itself.
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


@contextmanager
@require_jet_dependency("rich")
def _bs_pretty_progress() -> Iterator[None]:
    from rich.console import Console
    from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeElapsedColumn, TimeRemainingColumn

    progress_re = re.compile(r"progress\s+(\d+(?:\.\d+)?)%")

    class ProgressFromLogHandler(Handler):
        def __init__(self, progress: Progress, task_id: TaskID, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.setLevel(kwargs.get("level", INFO))
            self.progress = progress
            self.task_id = task_id

        @contextmanager
        def with_logger(self, logger: Logger) -> Iterator[None]:
            logger.addHandler(self)

            try:
                yield
            finally:
                logger.removeHandler(self)

        def emit(self, record: LogRecord) -> None:
            try:
                m = progress_re.search(record.getMessage())
                if m:
                    pct = float(m.group(1))

                    self.progress.update(self.task_id, completed=pct, visible=True)
                    return
            except Exception:
                self.handleError(record)

    vs_logger = getLogger("vapoursynth")
    vs_logger.propagate = False

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=Console(stderr=True),
        transient=True,
    )

    task_id = progress.add_task("Indexing with BestSource...", total=100, visible=False)

    try:
        with progress, ProgressFromLogHandler(progress=progress, task_id=task_id, level=INFO).with_logger(vs_logger):
            yield
            progress.update(task_id, visible=True)
    finally:
        vs_logger.propagate = True
