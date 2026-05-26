from __future__ import annotations

from collections.abc import Iterable

from jetpytools import CustomNotImplementedError, FuncExcept, cachedproperty, to_arr

from ..exceptions import UnsupportedColorFamilyError, UnsupportedVideoFormatError
from ..types import HoldsVideoFormat, Planes, VideoFormatLike
from ..utils import check_variable, get_color_family, normalize_planes
from ..vs_proxy import VSObject, vs
from .utils import depth, join, plane

__all__ = ["FunctionUtil"]


class FunctionUtil(VSObject):
    """
    Utility class to standardize and simplify common boilerplate used in VapourSynth function wrappers.
    """

    def __init__(
        self,
        clip: vs.VideoNode,
        func: FuncExcept,
        planes: Planes = None,
        color_family: VideoFormatLike
        | HoldsVideoFormat
        | vs.ColorFamily
        | Iterable[VideoFormatLike | HoldsVideoFormat | vs.ColorFamily]
        | None = None,
        bitdepth: int | range | tuple[int, int] | set[int] | None = None,
    ) -> None:
        """
        Initializes the class.

        Args:
            clip: The clip to process.
            func: The function returned for custom error handling.
            planes: The planes that get processed in the function. Default: All planes.
            color_family: The accepted color families.
                If the clip does not match one of these, an `UnsupportedColorFamilyError` is raised.
                Default: All color families.
            bitdepth: Specifies acceptable bit depths for the input clip. Can be:

                   - ``int``: a single allowed bit depth.
                   - ``set``: multiple allowed bit depths.
                   - ``tuple`` or ``range``: inclusive range of allowed bit depths.
                   - ``None``: no bit depth restriction (no normalization).

                If the clip's bit depth is below the allowed range, it will be raised to the minimum allowed.
                If the bit depth exceeds the maximum allowed, an `UnsupportedVideoFormatError` is raised.
                No automatic down-conversion occurs when returning the processed clip
                via [return_clip][vstools.FunctionUtil.return_clip].
        """
        assert check_variable(clip, func)

        self.clip = clip
        self.func = func

        if color_family is None:
            color_family = [vs.GRAY, vs.RGB, vs.YUV]

        self.allowed_cfamilies = frozenset(get_color_family(c) for c in to_arr(color_family))  # type: ignore[arg-type]

        UnsupportedColorFamilyError.check(self.clip, self.allowed_cfamilies, func)

        if self.allowed_cfamilies == {vs.GRAY}:
            planes = 0

        self.planes = normalize_planes(self.clip, planes)

        if self.clip.format.color_family is vs.RGB and self.planes != [0, 1, 2]:
            raise CustomNotImplementedError

        if isinstance(bitdepth, int):
            bitdepth = {bitdepth}
        elif isinstance(bitdepth, tuple):
            bitdepth = range(bitdepth[0], bitdepth[1] + 1)
        elif bitdepth is None:
            bitdepth = set()

        self.allowed_bitdepth = frozenset(bitdepth)

    @cachedproperty
    def norm_clip(self) -> vs.VideoNode:
        """
        The input clip normalized to the allowed bit depth.

        Raises:
            UnsupportedVideoFormatError: If the input clip has a bit depth higher than the allowed maximum.
        """
        if not self.allowed_bitdepth or self.clip.format.bits_per_sample in self.allowed_bitdepth:
            return self.clip

        bits = next((b for b in sorted(self.allowed_bitdepth) if b >= self.clip.format.bits_per_sample), None)

        if bits is not None:
            return depth(self.clip, bits)

        raise UnsupportedVideoFormatError(
            self.func,
            self.clip,
            (
                self.clip.format.replace(bits_per_sample=b, sample_type=vs.SampleType(b > 16))
                for b in sorted(self.allowed_bitdepth)
            ),
        )

    @cachedproperty
    def work_clip(self) -> vs.VideoNode:
        """
        The working clip to be processed based on selected planes.
        """
        return plane(self.norm_clip, 0) if self.luma_only else self.norm_clip

    @cachedproperty
    def chroma_planes(self) -> list[vs.VideoNode]:
        """
        The chroma planes extracted from the normalized clip.
        """
        if self.planes != [0] or self.norm_clip.format.num_planes == 1 or self.clip.format.color_family is vs.RGB:
            return []

        return [plane(self.norm_clip, i) for i in (1, 2)]

    @property
    def luma(self) -> bool:
        """
        Whether the luma (Y) plane is included in processing.
        """
        return 0 in self.planes and self.clip.format.color_family in (vs.GRAY, vs.YUV)

    @property
    def luma_only(self) -> bool:
        """
        Whether only the luma (Y) plane is processed.
        """
        return self.planes == [0] and self.clip.format.color_family in (vs.GRAY, vs.YUV)

    @property
    def chroma(self) -> bool:
        """
        Whether any chroma (U/V) planes are included in processing.
        """
        return any(p in self.planes for p in [1, 2]) and self.clip.format.color_family is vs.YUV

    @property
    def chroma_only(self) -> bool:
        """
        Whether only chroma (U/V) planes are processed.
        """
        return self.planes == [1, 2] and self.clip.format.color_family is vs.YUV

    @property
    def chroma_pplanes(self) -> list[int]:
        """
        The list of chroma plane indices being processed.
        """
        chroma_pplanes = self.planes.copy()

        try:
            chroma_pplanes.remove(0)
            return chroma_pplanes
        except ValueError:
            return chroma_pplanes

    def return_clip(self, processed: vs.VideoNode, prop_src: vs.VideoNode | None = None) -> vs.VideoNode:
        """
        Recombine the processed clip with chroma or luma as needed and return the final output.

        If only the luma was processed, the original chroma is merged back.
        If only the chroma was processed, the original luma is merged back.

        Args:
            processed: The clip after processing.
            prop_src: Optional clip to copy frame properties from.

        Returns:
            The fully reconstructed clip with appropriate planes merged.
        """
        if self.chroma_planes:
            processed = join([processed, *self.chroma_planes], self.norm_clip.format.color_family, prop_src=prop_src)

        if self.chroma_only:
            processed = join(self.norm_clip, processed, prop_src=prop_src)

        return processed
