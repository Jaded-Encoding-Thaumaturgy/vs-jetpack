from __future__ import annotations

import re
from collections.abc import Callable, Generator
from contextlib import contextmanager, nullcontext
from logging import Handler, Logger, LogRecord, getLogger
from typing import TYPE_CHECKING, Any, Literal, override

from jetpytools import CustomIntEnum, SPathLike

from vsjetpack import deprecated
from vstools import core, vs

if TYPE_CHECKING:
    from rich.console import Console
    from rich.progress import Progress

from .base import CacheIndexer, Indexer

__all__ = ["FFMS2", "LSMAS", "BestSource", "ZipSource"]


class _ProgressFromLogHandler(Handler):
    def __init__(self, cb: Callable[[float], Any]) -> None:
        super().__init__()
        self.cb = cb
        self.progress_re = re.compile(r"progress\s+(\d+(?:\.\d+)?)%")

    @override
    def emit(self, record: LogRecord) -> None:
        try:
            m = self.progress_re.search(record.getMessage())
            if m:
                self.cb(float(m.group(1)))
                return
        except Exception:
            self.handleError(record)

    @contextmanager
    def with_logger(self, logger: Logger) -> Generator[None]:
        logger.addHandler(self)
        try:
            yield
        finally:
            logger.removeHandler(self)


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
        show_pretty_progress: bool | Callable[[float], None] | None = False,
        **kwargs: Any,
    ) -> None:
        """

        Note:
            You will need to call [setup_logging][vsjetpack.setup_logging] or [basicConfig][logging.basicConfig]
            to show progress.

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

        if not (p := kwargs.pop("show_pretty_progress")):
            return super().source_func(path, **kwargs)

        with cls.pretty_progress(p):
            return super().source_func(path, **kwargs)

    @classmethod
    @contextmanager
    def pretty_progress(cls, progress: Literal[True] | Callable[[float], None]) -> Generator[None]:
        if callable(progress):
            pr_ctx = nullcontext()
            cb = progress
        else:
            pr_ctx = cls.get_progress()
            task_id = pr_ctx.add_task("Indexing with BestSource...", total=100.0, visible=False)
            cb = lambda pct: pr_ctx.update(task_id, completed=pct, visible=True)  # noqa: E731

        handler = _ProgressFromLogHandler(cb)

        vs_logger = getLogger("vapoursynth")
        vs_logger.propagate = False

        try:
            with pr_ctx, handler.with_logger(vs_logger):
                yield
                cb(100.0)
        finally:
            vs_logger.propagate = True

    @staticmethod
    def get_progress(*, console: Console | None = None) -> Progress:
        from rich.console import Console
        from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

        return Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console or Console(stderr=True),
            transient=True,
        )


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


@deprecated(
    "LSMAS is deprecated and will be removed in a future version once FFMS2 fully supports common formats. "
    "BestSource generally provides the fastest seeking, FFMS2 the fastest indexing, "
    "and LSMAS may still be preferable for some formats such as .m2ts.",
    category=PendingDeprecationWarning,
)
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


class ZipSource(Indexer):
    """
    vszip image reader indexer
    """

    _source_func = core.lazy.vszip.ImageRead
