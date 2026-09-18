import asyncio
import hashlib
import os
import shutil
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from logging import ERROR, INFO, WARNING, getLogger
from pathlib import Path
from types import TracebackType
from typing import Annotated, Any, Self

import cyclopts
import cyclopts.help
import humanize
import niquests
import questionary as quest
from cyclopts.help import HelpPanel
from rich.console import Console, ConsoleOptions
from rich.pretty import pretty_repr
from rich.progress import BarColumn, DownloadColumn, Progress, TaskID, TextColumn, TransferSpeedColumn
from rich.text import Text

from vsjetpack import __version__

from .feeds import Asset, Feed, Release
from .settings import TOML_CONFIG, TOML_KEYS, get_artifacts_folder, get_onnx_folder, get_toml_config, write_toml_config

MAX_CONCURRENCY = os.cpu_count() or 4

logger = getLogger(__name__)


def _custom_help_formatter(console: Console, options: ConsoleOptions, panel: HelpPanel) -> None:
    for i, entry in enumerate(panel.entries):
        if "--provider" in entry.positive_names:
            clean_names = tuple(name for name in entry.positive_names if name != "--provider")
            panel.entries[i] = entry.copy(positive_names=clean_names)
    cyclopts.help.DefaultFormatter()(console, options, panel)


app = cyclopts.App(
    name="vsscale",
    version=__version__,
    help="CLI utility for managing machine learning models and TensorRT/MIGraphX artifacts for VapourSynth.",
    help_on_error=True,
    console=Console(stderr=True),
    config=[
        cyclopts.config.Env("VSSCALE_"),
        cyclopts.config.Toml(TOML_CONFIG[0], root_keys=TOML_KEYS[0], allow_unknown=True),
        cyclopts.config.Toml(TOML_CONFIG[1], root_keys=TOML_KEYS[1], allow_unknown=True),
    ],
    help_formatter=_custom_help_formatter,
)
onnx_app = cyclopts.App(name="onnx", help="Manage downloaded ONNX models.")
artifact_app = cyclopts.App(name="artifact", help="Manage built TensorRT and MIGraphxX artifacts.")
config_app = cyclopts.App(name="config", help="Manage vsscale configuration.")
app.command(onnx_app)
app.command(artifact_app)
app.command(config_app)


@app.meta.default
def meta_main(
    *tokens: Annotated[str, cyclopts.Parameter(show=False, allow_leading_hyphen=True)],
    no_config: Annotated[
        bool,
        cyclopts.Parameter(
            negative=(),
            show_default=False,
            help="Ignore TOML configuration files and environment variables.",
        ),
    ] = False,
) -> None:
    os.environ["VSSCALE_CLI"] = "1"
    if no_config:
        app.config = None
    try:
        app(tokens)
    except Exception as e:  # noqa: BLE001
        app.console.print(f"[red]{e.__class__.__name__}:[/red] {e}")
        raise SystemExit(1)


@onnx_app.command(help_formatter=_custom_help_formatter)
async def download(
    *provider: Annotated[str, cyclopts.Parameter(name="--provider")],
    latest: Annotated[
        bool,
        cyclopts.Parameter(negative=(), show_default=False, env_var="VSSCALE_LATEST"),
    ] = False,
    global_: Annotated[
        bool,
        cyclopts.Parameter(negative=(), show_default=False, env_var="VSSCALE_GLOBAL"),
    ] = False,
    assumeyes: Annotated[
        bool,
        cyclopts.Parameter(alias="-y", negative=(), show_default=False),
    ] = False,
    console: Annotated[Console | None, cyclopts.Parameter(parse=False)] = None,
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
        provider: The ONNX model(s) to download. Possible choices: "ArtCNN", "DPIR", "Waifu2X".
            Use '==' syntax to pin a version (e.g. ArtCNN==v1.6.2).
        latest: Whether to automatically download all models from the latest release.
        global_: Whether to download models to the global folder.
        assumeyes: Answer yes for all questions.
    """
    if not provider:
        # Fully interactive: pick model, then tag, then assets
        feed = await _select_model()
        releases = await _fetch_releases(feed)
        release = await _select_tag(releases)
        assets = await _select_assets(release)
        dest_folder = get_onnx_folder(global_=global_) / feed.display_name.lower() / release.tag
        if not assumeyes:
            await _confirm_download(dest_folder)
        return await _download_assets(feed, assets, dest_folder)

    for spec in provider:
        model_name, pinned_version = _parse_model_spec(spec)
        feed = _find_feed(model_name)

        releases = await _fetch_releases(feed, console=console)

        if pinned_version is not None:
            release = next((r for r in releases if r.tag == pinned_version), None)

            if not release:
                raise ValueError(
                    f"Version {pinned_version} not found. Available versions: {', '.join(r.tag for r in releases[:10])}"
                )

            assets = release.assets
        elif latest:
            release = releases[0]
            assets = release.assets
            _display(
                INFO,
                "[bold]Latest release for %s: %s (%s)[/bold]",
                feed.display_name,
                release.tag,
                release.published_at[:10],
            )
        else:
            release = await _select_tag(releases)
            assets = await _select_assets(release)

        dest_folder = get_onnx_folder(global_=global_) / feed.display_name.lower() / release.tag

        if not assumeyes:
            await _confirm_download(dest_folder)
        await _download_assets(feed, assets, dest_folder, console=console)
        _display(INFO, "")


@artifact_app.command(help="List built TensorRT & MIGraphxX artifacts.")
@onnx_app.command(help="List downloaded ONNX models.")
def show(
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

    files = (f for f in folder.glob("**/*", case_sensitive=False) if f.suffix in ext)
    return print(pretty_repr(sorted(files, reverse=True)))


@artifact_app.command(help="Clear built TensorRT & MIGraphxX artifacts.")
@onnx_app.command(help="Clear downloaded ONNX models.")
def clear(
    global_: Annotated[
        bool,
        cyclopts.Parameter(negative=(), show_default=False, env_var=["VSSCALE_CLEAR_GLOBAL", "VSSCALE_GLOBAL"]),
    ] = False,
) -> None:
    """
    Delete downloaded ONNX models or built TensorRT & MIGraphxX artifacts.

    If no model specs are provided, the entire directory is cleared.

    Args:
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

    return shutil.rmtree(folder, ignore_errors=True)


@config_app.default
def config(
    file: Path | None = None,
    /,
    *provider: str,
    latest: bool | None = None,
    auto: bool | None = None,
    global_: bool | None = None,
    fallback: bool | None = None,
    assumeyes: Annotated[bool, cyclopts.Parameter(alias="-y", negative=(), show_default=False)] = False,
) -> None:
    """
    Write or update vsscale configuration in pyproject.toml or vsjet.toml.

    Args:
        file: Target configuration file ('pyproject.toml' or 'vsjet.toml').
        provider: Default ONNX model(s) to configure (e.g. 'ArtCNN', 'DPIR', 'Waifu2x').
        latest: Whether to download latest model releases by default.
        auto: Whether to automatically download missing models on demand.
        global_: Whether to use global cache directory by default.
        fallback: Whether to enable global cache fallback.
        assumeyes: Answer yes for all questions and skip interactive wizard.
    """
    if file is None:
        toml = "pyproject.toml" if Path("pyproject.toml").exists() else "vsjet.toml"
        if assumeyes:
            target_path = Path(toml)
        else:
            choices = [
                quest.Choice("pyproject.toml", value="pyproject.toml"),
                quest.Choice("vsjet.toml", value="vsjet.toml"),
            ]
            selected = quest.select("Select configuration file target:", choices, toml, qmark="📝").ask()
            if selected is None:
                raise SystemExit(0)
            target_path = Path(selected)
    else:
        target_path = Path(file)

    if not assumeyes:
        if global_ is None:
            global_ = quest.confirm("Use global cache directory by default?", default=False, qmark="🌐").ask()
            if global_ is None:
                raise SystemExit(0)

        if fallback is None:
            fallback = quest.confirm(
                "Enable global cache fallback if local model is missing?", default=True, qmark="🔄"
            ).ask()
            if fallback is None:
                raise SystemExit(0)

        if not provider:
            choices = [quest.Choice(title=name, value=name, checked=True) for name in Feed.all_feeds]
            providers = quest.checkbox("Select default ONNX models to configure:", choices=choices, qmark="📦").ask()
            if providers is None:
                raise SystemExit(0)
            provider = tuple(providers)

        if latest is None:
            latest = quest.confirm("Download latest model release automatically?", default=True, qmark="🏷️").ask()
            if latest is None:
                raise SystemExit(0)

        if auto is None:
            auto = quest.confirm("Auto-download models when used in Python?", default=True, qmark="⚡").ask()
            if auto is None:
                raise SystemExit(0)

    written_path = write_toml_config(
        target_path,
        global_=global_,
        fallback=fallback,
        provider=provider,
        latest=latest,
        auto=auto,
    )

    _display(INFO, "[green]✔ Successfully updated configuration in [bold]%s[/bold][/green]", written_path.name)


@config_app.command(name="show")
def show_config() -> None:
    """Display the currently active vsscale configuration."""
    config = get_toml_config()
    detected_file = None
    for config_file in TOML_CONFIG:
        p = Path(config_file).expanduser().resolve().absolute()
        if p.exists():
            detected_file = p
            break

    if detected_file:
        _display(INFO, "[bold]Active configuration file:[/bold] [cyan]%s[/cyan]", detected_file)
    else:
        _display(WARNING, "No active configuration file found.")

    _display(INFO, pretty_repr(config))


def _parse_model_spec(spec: str) -> tuple[str, str | None]:
    if "==" not in spec:
        return spec.strip().lower(), None

    name, _, version = spec.partition("==")
    return name.strip().lower(), version.strip()


def _find_feed(name: str) -> Feed:
    lookup = {k.lower(): k for k in Feed.all_feeds}

    if (matched_key := lookup.get(name)) is None:
        raise ValueError(f"Unknown model '{name}'. Available models: {', '.join(Feed.all_feeds)}")

    return Feed.all_feeds[matched_key]()


async def _fetch_releases(feed: Feed, console: Console | None = None) -> list[Release]:
    console = console or app.console
    try:
        with console.status(f"Fetching releases for [bold]{feed.display_name}[/bold]...", spinner="dots"):
            async with niquests.AsyncSession(disable_http3=True) as session:
                releases = await feed.releases(session)
    except niquests.exceptions.HTTPError as e:
        match e.response:
            case None:
                raise
            case res if res.status_code == 401:
                raise OSError(
                    "GitHub API token is unauthorized (401). "
                    "Please check if your GITHUB_TOKEN environment variable is correct and valid."
                )
            case res if res.status_code == 403:
                raise OSError(
                    "GitHub API rate limit exceeded or access forbidden (403). "
                    "Please try again later or set the GITHUB_TOKEN environment variable to authenticate."
                )
            case _:
                raise

    if not releases:
        _display(WARNING, "No releases found for %s.", feed.display_name)
        raise SystemExit(0)

    return releases


# CLI Exclusive function
async def _select_model() -> Feed:
    choices = [quest.Choice(title=f"{name}", value=name) for name, _ in Feed.all_feeds.items()]

    selected = await quest.select("Select an ONNX provider to download:", choices=choices, qmark="📦").ask_async()

    if selected is None:
        raise SystemExit(0)

    return Feed.all_feeds[selected]()


# CLI Exclusive function
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


# CLI Exclusive function
async def _select_assets(release: Release) -> list[Asset]:
    choices = [quest.Choice(f"{a.name}  ({humanize.naturalsize(a.size)})", a, checked=True) for a in release.assets]

    msg = f"Select models to download from {release.tag}:"
    selected = await quest.checkbox(msg, choices, qmark="📥").ask_async()

    if selected is None or len(selected) == 0:
        app.console.print("[yellow]No models selected. Aborting.[/yellow]")
        raise SystemExit(0)

    return selected


# CLI Exclusive function
async def _confirm_download(dest_folder: Path) -> None:
    msg = f"The models will be downloaded to: '{dest_folder}'"
    res = await quest.confirm(msg).ask_async()

    if not res:
        raise SystemExit(1)


async def _download_assets(
    feed: Feed,
    assets: Sequence[Asset],
    dest_folder: Path,
    console: Console | None = None,
) -> None:
    _display(INFO, "[bold]Downloading %s to:[/bold] [cyan]%s[/cyan]", feed.display_name, dest_folder)

    async with niquests.AsyncSession(
        pool_connections=MAX_CONCURRENCY,
        pool_maxsize=MAX_CONCURRENCY,
        disable_http3=True,
    ) as session:
        dest_folder.mkdir(parents=True, exist_ok=True)
        downloader = _AssetDownloader(feed, dest_folder, session, console=console or app.console)
        await downloader.download(assets)

    if downloader.downloaded:
        dl_count = len(downloader.downloaded)
        _display(
            INFO,
            "[green]✔️  Downloaded %d %s model%s.[/green]",
            dl_count,
            feed.display_name,
            "s" if dl_count != 1 else "",
        )
    if downloader.skipped:
        skip_count = len(downloader.skipped)
        _display(
            INFO,
            "[dim]⏭  Skipped %d already-downloaded %s model%s.[/dim]",
            skip_count,
            feed.display_name,
            "s" if skip_count != 1 else "",
        )
    if downloader.failed:
        fail_count = len(downloader.failed)
        _display(
            ERROR,
            "[red]❌  Failed to download %d %s model%s.[/red]",
            fail_count,
            feed.display_name,
            "s" if fail_count != 1 else "",
        )


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
    dest_folder: Path
    session: niquests.AsyncSession
    console: Console

    def __post_init__(self) -> None:
        self.sema = asyncio.Semaphore(MAX_CONCURRENCY)
        self.skipped = set[Asset]()
        self.downloaded = set[Asset]()
        self.failed = dict[Asset, Exception]()
        self.progress = _AsyncProgress(
            TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
            BarColumn(bar_width=40),
            "[progress.percentage]{task.percentage:>3.0f}%",
            DownloadColumn(),
            TransferSpeedColumn(),
            console=self.console,
            refresh_per_second=30,
        )

    async def download(self, assets: Sequence[Asset]) -> None:
        async with self.progress, asyncio.TaskGroup() as tg:
            for asset in assets:
                tg.create_task(self._download_asset(asset), name=asset.name)

    async def _download_asset(self, asset: Asset) -> None:
        dest_path = self.dest_folder / asset.name

        def get_hash() -> str:
            with dest_path.open("rb") as f:
                return hashlib.file_digest(f, "sha256").hexdigest()

        if dest_path.exists() and get_hash() == asset.sha256:
            _display(INFO, "  [dim]⏭ %s: %s (already downloaded)[/dim]", self.feed.display_name, asset.name)
            self.skipped.add(asset)
            return

        task: TaskID | None = None
        try:
            async with self.sema, _delete_on_error(dest_path):
                task = self.progress.add_task("download", filename=asset.name, total=asset.size)
                hasher = hashlib.sha256()

                res = await self.session.get(asset.url, stream=True, headers=self.feed.headers)

                async with res.raise_for_status():
                    with dest_path.open("wb") as f:
                        async for chunk in await res.iter_content(chunk_size=64 * 1024):
                            f.write(chunk)
                            hasher.update(chunk)
                            self.progress.update(task, advance=len(chunk))
                    self.progress.update(task, visible=False)

                if (computed_hash := hasher.hexdigest()) != asset.sha256:
                    raise ValueError(
                        f"Integrity check failed for {asset.name}. "
                        f"Expected sha256: {asset.sha256}, got: {computed_hash}"
                    )

                self.downloaded.add(asset)
        except (niquests.RequestException, ValueError, OSError) as e:
            if task is not None:
                self.progress.update(task, visible=False)
            self.failed[asset] = e
            _display(ERROR, "  [red]❌ %s: %s (%s)[/red]", self.feed.display_name, asset.name, e)


@asynccontextmanager
async def _delete_on_error(dest_path: Path) -> AsyncGenerator[None]:
    try:
        yield
    except Exception:
        dest_path.unlink(missing_ok=True)
        raise


def _display(level: int, msg: str, *args: object, **kwargs: Any) -> None:
    if os.environ.get("VSSCALE_CLI") == "1":
        if level == WARNING:
            msg = f"[yellow]{msg}[/yellow]"
        elif level == ERROR:
            msg = f"[red]{msg}[/red]"
        app.console.print(msg % args if args else msg)
    elif msg:
        logger.log(level, Text.from_markup(msg).plain, *args, **kwargs)
