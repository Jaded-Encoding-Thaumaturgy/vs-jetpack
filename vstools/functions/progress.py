from __future__ import annotations

from types import TracebackType
from typing import overload

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    Task,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text
from typing_extensions import Self

__all__ = [
    "BarColumn",
    "FPSColumn",
    "Progress",
    "RenderProgressCTX",
    "TextColumn",
    "TimeRemainingColumn",
    "get_render_progress",
]


class FPSColumn(ProgressColumn):
    """
    Progress rendering.
    """

    def render(self, task: Task) -> Text:
        """
        Render bar.
        """

        return Text(f"{task.speed or 0:.02f} fps")


class RenderProgressCTX:
    def __init__(self, progress: Progress, task_id: TaskID) -> None:
        self.progress = progress
        self.task_id = task_id

    def __enter__(self) -> Self:
        self.progress.__enter__()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        self.progress.__exit__(exc_type, exc_val, exc_tb)

    @overload
    def update(self) -> None: ...

    @overload
    def update(self, *, advance: int) -> None: ...

    @overload
    def update(self, completed: int, total: int) -> None: ...

    def update(self, completed: int | None = None, total: int | None = None, advance: int = 1) -> None:
        return self.progress.update(self.task_id, completed=completed, total=total, advance=advance)


@overload
def get_render_progress() -> Progress: ...


@overload
def get_render_progress(title: str, total: int) -> RenderProgressCTX: ...


def get_render_progress(title: str | None = None, total: int | None = None) -> RenderProgressCTX | Progress:
    """
    Return render progress.
    """

    if title and total:
        progress = get_render_progress()

        return RenderProgressCTX(progress, progress.add_task(title, True, total))

    return Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("{task.percentage:>3.02f}%"),
        FPSColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=Console(stderr=True),
    )
