from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from logging import getLogger
from typing import Any, ClassVar, Literal, overload

from vstools import vs

type Shape = tuple[int, int]

logger = getLogger(__name__)


@dataclass(kw_only=True, frozen=True)
class Backend:
    """
    Base wrapper for a MLRT VapourSynth backend plugin.
    """

    plugin: ClassVar[vs.Plugin]
    flexible_output_prop: ClassVar[str] = "MlrtFlexible"

    class OutputFormat(IntEnum):
        """
        Output format for the backend plugin.
        """

        FP32 = 0
        FP16 = 1

    @overload
    def inference(
        self,
        clips: vs.VideoNode | Sequence[vs.VideoNode],
        network_path: str | os.PathLike[str],
        /,
        overlap: tuple[int, int],
        tilesize: tuple[int, int],
        *,
        flexible: Literal[False] = ...,
        **kwargs: Any,
    ) -> vs.VideoNode: ...

    @overload
    def inference(
        self,
        clips: vs.VideoNode | Sequence[vs.VideoNode],
        network_path: str | os.PathLike[str],
        /,
        overlap: tuple[int, int],
        tilesize: tuple[int, int],
        *,
        flexible: Literal[True],
        **kwargs: Any,
    ) -> list[vs.VideoNode]: ...

    @overload
    def inference(
        self,
        clips: vs.VideoNode | Sequence[vs.VideoNode],
        network_path: str | os.PathLike[str],
        /,
        overlap: tuple[int, int],
        tilesize: tuple[int, int],
        *,
        flexible: bool = ...,
        **kwargs: Any,
    ) -> vs.VideoNode | list[vs.VideoNode]: ...

    def inference(
        self,
        clips: vs.VideoNode | Sequence[vs.VideoNode],
        network_path: str | os.PathLike[str],
        /,
        overlap: tuple[int, int],
        tilesize: tuple[int, int],
        *,
        flexible: bool = False,
        **kwargs: Any,
    ) -> vs.VideoNode | list[vs.VideoNode]:
        """
        Run inference with this backend.

        Args:
            clips: Input clip or clips passed to the backend model.
            network_path: Path to the model file or backend artifact.
            overlap: Horizontal and vertical tile overlap in pixels.
            tilesize: Horizontal and vertical tile size in pixels.
            flexible: Return each flexible output plane as a separate clip.
            **kwargs: Additional backend plugin arguments forwarded unchanged.

        Returns:
            A single output clip, or a list of output clips when `flexible` is enabled.
        """
        args = self.get_args(clips)

        if flexible:
            args = args.copy()
            args["flexible_output_prop"] = self.flexible_output_prop

        logger.info("Calling %s.Model", self.plugin.namespace)
        logger.info("Clips: %r", clips)
        logger.info("Network Path: %s", network_path)
        logger.info("overlap=%s, tilesize=%s, %s", overlap, tilesize, args | kwargs)
        output = self.plugin.Model(clips, network_path, overlap, tilesize, **args | kwargs)

        if flexible:
            clip = output["clip"]
            num_planes = output["num_planes"]

            output = [clip.std.PropToClip(prop=f"{self.flexible_output_prop}{i}") for i in range(num_planes)]

        return output

    def get_args(self, clips: vs.VideoNode | Sequence[vs.VideoNode]) -> dict[str, Any]:
        """Return backend plugin arguments derived from this configuration."""
        return {}
