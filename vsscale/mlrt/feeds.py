from __future__ import annotations

import os
from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, ClassVar, NewType

import niquests
from jetpytools import classproperty

logger = getLogger(__name__)

GitTag = NewType("GitTag", str)
URL = NewType("URL", str)


@dataclass(frozen=True, slots=True)
class Asset:
    name: str
    url: URL
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class Release:
    tag: GitTag
    published_at: str
    assets: Sequence[Asset] = field(default_factory=list)


class Feed(ABC):
    """Base class for model feeds backed by GitHub releases."""

    display_name: ClassVar[str]
    base_url: ClassVar[str]
    _registry: ClassVar[dict[str, type[Feed]]] = {}

    def __init_subclass__(cls) -> None:
        cls._registry[cls.display_name] = cls

    @property
    def headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token := os.environ.get("GITHUB_TOKEN"):
            headers["Authorization"] = f"Bearer {token}"

        return headers

    @classproperty.cached
    @classmethod
    def all_feeds(cls) -> dict[str, type[Feed]]:
        return dict(cls._registry)

    async def releases(self, session: niquests.AsyncSession) -> list[Release]:
        raw_releases = list[dict[str, Any]]()

        page = 1
        while True:
            batch = (await session.get(self.base_url.format(page=page), headers=self.headers)).raise_for_status().json()
            if not batch:
                break
            raw_releases.extend(batch)
            page += 1

        result = list[Release]()
        for r in raw_releases:
            tag = GitTag(r["tag_name"])
            published = r.get("published_at", "")
            raw_assets = r.get("assets", [])

            if not raw_assets:
                continue

            onnx_assets = [
                Asset(
                    name=a["name"],
                    url=URL(a["browser_download_url"]),
                    size=a["size"],
                    sha256=a["digest"].removeprefix("sha256:"),
                )
                for a in raw_assets
                if a["name"].endswith(".onnx") and a.get("digest")
            ]

            if onnx_assets:
                result.append(Release(tag, published, onnx_assets))

        return result


class ArtCNNFeed(Feed):
    display_name = "ArtCNN"
    repo = "Artoriuz/ArtCNN"
    base_url = f"https://api.github.com/repos/{repo}/releases?per_page=100&page={{page}}"


class Waifu2XFeed(Feed):
    display_name = "Waifu2X"
    repo = "Jaded-Encoding-Thaumaturgy/nunif"
    base_url = f"https://api.github.com/repos/{repo}/releases?per_page=100&page={{page}}"


class DPIRFeed(Feed):
    display_name = "DPIR"
    repo = "Jaded-Encoding-Thaumaturgy/KAIR"
    base_url = f"https://api.github.com/repos/{repo}/releases?per_page=100&page={{page}}"
