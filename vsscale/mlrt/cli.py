import hashlib
import os
import shutil
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import TracebackType
from typing import Annotated, Self

import anyio
import anyio.to_thread
import cyclopts
import cyclopts.help
import humanize
import niquests
import questionary as quest
from cyclopts.help import HelpPanel
from rich.console import Console, ConsoleOptions
from rich.pretty import pretty_repr
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TransferSpeedColumn

from vsjetpack import __version__

from .feeds import Asset, Feed, Release
from .settings import TOML_CONFIG, TOML_KEYS, get_artifacts_folder, get_onnx_folder

MAX_CONCURRENCY = os.cpu_count() or 4

console = Console(stderr=True)

app = cyclopts.App(
    name="vsscale",
    version=__version__,
    help="CLI utility for managing machine learning models and TensorRT/MIGraphX artifacts for VapourSynth.",
    help_on_error=True,
    console=console,
    config=[
        cyclopts.config.Env("VSSCALE_"),
        cyclopts.config.Toml(TOML_CONFIG[0], root_keys=TOML_KEYS[0]),
        cyclopts.config.Toml(TOML_CONFIG[1], root_keys=TOML_KEYS[1]),
    ],
)
onnx_app = cyclopts.App(name="onnx", help="Manage downloaded ONNX models.")
artifact_app = cyclopts.App(name="artifact", help="Manage built TensorRT and MIGraphxX artifacts.")
app.command(onnx_app)
app.command(artifact_app)


@app.meta.default
def meta_main(
    *tokens: Annotated[str, cyclopts.Parameter(show=False)],
    no_config: Annotated[
        bool,
        cyclopts.Parameter(
            negative=(),
            show_default=False,
            help="Ignore TOML configuration files and environment variables.",
        ),
    ] = False,
) -> None:
    if no_config:
        app.config = None
    app(tokens)


def _custom_help_formatter(console: Console, options: ConsoleOptions, panel: HelpPanel) -> None:
    for i, entry in enumerate(panel.entries):
        if "--onnx" in entry.positive_names:
            clean_names = tuple(name for name in entry.positive_names if name != "--onnx")
            panel.entries[i] = entry.copy(positive_names=clean_names)  # type: ignore[no-untyped-call]
    cyclopts.help.DefaultFormatter()(console, options, panel)


@onnx_app.command(help_formatter=_custom_help_formatter)
async def download(
    *onnx: Annotated[str, cyclopts.Parameter(name="--onnx")],
    latest: Annotated[
        bool,
        cyclopts.Parameter(
            negative=(),
            show_default=False,
            env_var=["VSSCALE_ONNX_DOWNLOAD_LATEST", "VSSCALE_LATEST"],
        ),
    ] = False,
    global_: Annotated[
        bool,
        cyclopts.Parameter(
            negative=(),
            show_default=False,
            env_var=["VSSCALE_ONNX_DOWNLOAD_GLOBAL", "VSSCALE_GLOBAL"],
        ),
    ] = False,
) -> None:
    """
    Download ONNX models.

    Supports multiple invocation styles:
      - Interactive:        vsscale onnx download
      - Pick tag for model: vsscale onnx download ArtCNN
      - Pinned version:     vsscale onnx download ArtCNN==v1.6.2
      - Latest release:     vsscale onnx download ArtCNN --latest

    If a `vsjet.toml` or a `pyproject.toml` file is detected with a valid configuration,
    the interactive mode may be partially or fully skipped.

    Args:
        onnx: The ONNX model(s) to download. Possible choices: "ArtCNN", "DPIR", "Waifu2X".
            Use '==' syntax to pin a version (e.g. ArtCNN==v1.6.2).
        latest: Whether to automatically download all models from the latest release.
        global_: Whether to download models to the global folder.
    """
    if not onnx:
        # Fully interactive: pick model, then tag, then assets
        feed = await _select_model()
        releases = await _fetch_releases(feed)
        release = await _select_tag(releases)
        assets = await _select_assets(release)
        return await _download_assets(feed, release, assets, global_=global_)

    for spec in onnx:
        model_name, pinned_version = _parse_model_spec(spec)
        feed = _find_feed(model_name)

        releases = await _fetch_releases(feed)

        if pinned_version is not None:
            release = next((r for r in releases if r.tag == pinned_version), None)

            if not release:
                console.print(f"[red]Error: Version '{pinned_version}' not found.[/red]")
                available_tags = ", ".join(r.tag for r in releases[:10])
                console.print(f"[yellow]Available versions: {available_tags}[/yellow]")
                raise SystemExit(1)

            assets = release.assets
        elif latest:
            release = releases[0]
            assets = release.assets
            console.print(f"[bold]Latest release: {release.tag} ({release.published_at[:10]})[/bold]")
        else:
            release = await _select_tag(releases)
            assets = await _select_assets(release)

        await _download_assets(feed, release, assets, global_=global_)
        console.print()


@artifact_app.command(help="List built TensorRT & MIGraphxX artifacts.", help_formatter=_custom_help_formatter)
@onnx_app.command(help="List downloaded ONNX models.", help_formatter=_custom_help_formatter)
def show(
    *onnx: Annotated[str, cyclopts.Parameter(name="--onnx")],
    global_: Annotated[
        bool,
        cyclopts.Parameter(
            negative=(),
            show_default=False,
            env_var=["VSSCALE_SHOW_GLOBAL", "VSSCALE_GLOBAL"],
        ),
    ] = False,
) -> None:
    """
    List downloaded ONNX models or built TensorRT & MIGraphxX artifacts.

    Args:
        onnx: The model(s) to show. Supports specifying version pin (e.g. ArtCNN==v1.6.2).
            If not specified, all files are listed.
        global_: Whether to show models in the global folder.
    """
    (cmd, *_), _, _ = app.parse_commands()

    match cmd:
        case "onnx":
            folder = get_onnx_folder(global_=global_)
            ext = [".onnx"]
        case "artifact":
            folder = get_artifacts_folder(global_=global_)
            ext = [".mxr", ".engine", ".cache"]
        case _:
            raise ValueError

    if not onnx:
        files = (f for f in folder.glob("**/*", case_sensitive=False) if f.suffix in ext)
        return print(pretty_repr(sorted(files, reverse=True)))

    for spec in onnx:
        model_name, pinned_version = _parse_model_spec(spec)
        spec_folder = folder / model_name / (pinned_version or "")

        files = (f for f in spec_folder.glob("**/*", case_sensitive=False) if f.suffix in ext)
        folder_repr = pretty_repr(sorted(files, reverse=True))
        print(folder_repr)


@artifact_app.command(help="Clear built TensorRT & MIGraphxX artifacts.", help_formatter=_custom_help_formatter)
@onnx_app.command(help="Clear downloaded ONNX models.", help_formatter=_custom_help_formatter)
def clear(
    *onnx: Annotated[str, cyclopts.Parameter(name="--onnx")],
    global_: Annotated[
        bool,
        cyclopts.Parameter(negative=(), show_default=False, env_var=["VSSCALE_CLEAR_GLOBAL", "VSSCALE_GLOBAL"]),
    ] = False,
) -> None:
    """
    Delete downloaded ONNX models or built TensorRT & MIGraphxX artifacts.

    If no model specs are provided, the entire directory is cleared.

    Args:
        onnx: Specific model namespace(s) or model-version specification(s) to clear
            (e.g., "ArtCNN" or "ArtCNN==v1.6.2"). If omitted, all files will be deleted.
        global_: Whether to clear files in the global folder.
    """
    (cmd, *_), _, _ = app.parse_commands()

    match cmd:
        case "onnx":
            folder = get_onnx_folder(global_=global_)
        case "artifact":
            folder = get_artifacts_folder(global_=global_)
        case _:
            raise ValueError

    if not onnx:
        return shutil.rmtree(folder, ignore_errors=False)

    for spec in onnx:
        model_name, pinned_version = _parse_model_spec(spec)
        spec_folder = folder / model_name / (pinned_version or "")
        shutil.rmtree(spec_folder, ignore_errors=False)


def _parse_model_spec(spec: str) -> tuple[str, str | None]:
    if "==" not in spec:
        return spec.strip().lower(), None

    name, _, version = spec.partition("==")
    return name.strip().lower(), version.strip()


def _find_feed(name: str) -> Feed:
    lookup = {k.lower(): k for k in Feed.all_feeds}

    if (matched_key := lookup.get(name)) is None:
        console.print(f"[red]Error: Unknown model '{name}'.[/red]")
        console.print(f"[yellow]Available models: {', '.join(Feed.all_feeds)}[/yellow]")
        raise SystemExit(1)

    return Feed.all_feeds[matched_key]()


async def _fetch_releases(feed: Feed) -> list[Release]:
    try:
        with console.status(f"Fetching releases for [bold]{feed.display_name}[/bold]...", spinner="dots"):
            async with niquests.AsyncSession(disable_http3=True) as session:
                releases = await feed.releases(session)
    except niquests.exceptions.HTTPError as e:
        match e.response:
            case None:
                raise
            case res if res.status_code == 401:
                console.print("[red]Error: GitHub API token is unauthorized (401).[/red]")
                console.print(
                    "[yellow]Please check if your GITHUB_TOKEN environment variable is correct and valid.[/yellow]"
                )
            case res if res.status_code == 403:
                console.print("[red]Error: GitHub API rate limit exceeded or access forbidden (403).[/red]")
                console.print(
                    "[yellow]Please try again later or set the "
                    "GITHUB_TOKEN environment variable to authenticate.[/yellow]"
                )
            case _:
                raise
        raise SystemExit(1)

    if not releases:
        console.print("[yellow]No releases found for this model.[/yellow]")
        raise SystemExit(0)

    return releases


async def _select_model() -> Feed:
    choices = [quest.Choice(title=f"{name}", value=name) for name, _ in Feed.all_feeds.items()]

    selected = await quest.select("Select an ONNX model to download:", choices=choices, qmark="📦").ask_async()

    if selected is None:
        raise SystemExit(0)

    return Feed.all_feeds[selected]()


async def _select_tag(releases: list[Release]) -> Release:
    choices = [
        quest.Choice(
            title=f"{r.tag}  ({r.published_at[:10]}, {len(r.assets)} model{'s' if len(r.assets) != 1 else ''})",
            value=r,
        )
        for r in releases
    ]

    selected = await quest.select("Select a release version:", choices=choices, qmark="🏷️").ask_async()

    if selected is None:
        raise SystemExit(0)

    return selected


async def _select_assets(release: Release) -> list[Asset]:
    choices = [quest.Choice(f"{a.name}  ({humanize.naturalsize(a.size)})", a, checked=True) for a in release.assets]

    msg = f"Select models to download from {release.tag}:"
    selected = await quest.checkbox(msg, choices, qmark="📥").ask_async()

    if selected is None or len(selected) == 0:
        console.print("[yellow]No models selected. Aborting.[/yellow]")
        raise SystemExit(0)

    return selected


async def _download_assets(feed: Feed, release: Release, assets: Sequence[Asset], *, global_: bool = False) -> None:
    dest_folder = anyio.Path(get_onnx_folder(global_=global_) / feed.display_name.lower() / release.tag)

    console.print(f"[bold]Downloading to:[/bold] [cyan]{dest_folder}[/cyan]")

    async with niquests.AsyncSession(
        pool_connections=MAX_CONCURRENCY,
        pool_maxsize=MAX_CONCURRENCY,
        disable_http3=True,
    ) as session:
        await dest_folder.mkdir(parents=True, exist_ok=True)
        downloader = _AssetDownloader(feed, dest_folder, session)
        await downloader.download(assets)

    if downloader.downloaded:
        dl_count = len(downloader.downloaded)
        console.print(f"[green]✔️  Downloaded {dl_count} model{'s' if dl_count != 1 else ''}.[/green]")
    if downloader.skipped:
        skip_count = len(downloader.skipped)
        console.print(f"[dim]⏭  Skipped {skip_count} already-downloaded model{'s' if skip_count != 1 else ''}.[/dim]")


class _AsyncProgress(Progress):
    async def __aenter__(self) -> Self:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        return self.__exit__(exc_type, exc_val, exc_tb)


@dataclass
class _AssetDownloader:
    feed: Feed
    dest_folder: anyio.Path
    session: niquests.AsyncSession

    def __post_init__(self) -> None:
        self.limiter = anyio.CapacityLimiter(MAX_CONCURRENCY)
        self.skipped = set[Asset]()
        self.downloaded = set[Asset]()
        self.progress = _AsyncProgress(
            TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
            BarColumn(bar_width=40),
            "[progress.percentage]{task.percentage:>3.0f}%",
            DownloadColumn(),
            TransferSpeedColumn(),
            console=console,
        )

    async def download(self, assets: Sequence[Asset]) -> None:
        async with self.progress, anyio.create_task_group() as tg:
            for asset in assets:
                tg.start_soon(self._download_asset, asset, name=asset.name)

    async def _download_asset(self, asset: Asset) -> None:
        dest_path = self.dest_folder / asset.name

        if await dest_path.exists():
            data = await dest_path.read_bytes()
            hash_val = await anyio.to_thread.run_sync(lambda: hashlib.sha256(data).hexdigest())

            if hash_val == asset.sha256:
                self.progress.console.print(f"  [dim]⏭ {asset.name} (already downloaded)[/dim]")
                self.skipped.add(asset)
                return

        async with self.limiter, _delete_on_error(dest_path):
            task = self.progress.add_task("download", filename=asset.name, total=asset.size)

            async with (
                (await self.session.get(asset.url, stream=True, headers=self.feed.headers)).raise_for_status() as res,
                await anyio.open_file(dest_path, "wb") as f,
            ):
                async for chunk in await res.iter_content(chunk_size=64 * 1024):
                    await f.write(chunk)
                    self.progress.update(task, advance=len(chunk))
                self.progress.update(task, visible=False)
            self.downloaded.add(asset)


@asynccontextmanager
async def _delete_on_error(dest_path: anyio.Path) -> AsyncGenerator[None]:
    try:
        yield
    except Exception:
        shutil.rmtree(dest_path, ignore_errors=True)
        raise
