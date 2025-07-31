from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Generic, MutableMapping, TypeVar, cast

from jetpytools import T

from ..functions import Keyframes
from ..types import VideoNodeT, vs_object
from . import vs_proxy as vs

if TYPE_CHECKING:
    from vapoursynth._typings import _VapourSynthMapValue


__all__ = [
    "ClipFramesCache",
    "ClipsCache",
    "DynamicClipsCache",
    "FramesCache",
    "NodeFramesCache",
    "NodesPropsCache",
    "SceneBasedDynamicCache",
    "cache_clip",
]


NodeT = TypeVar("NodeT", bound=vs.RawNode)
FrameT = TypeVar("FrameT", bound=vs.RawFrame)


class ClipsCache(vs_object, dict[vs.VideoNode, vs.VideoNode]):
    def __delitem__(self, key: vs.VideoNode) -> None:
        if key not in self:
            return

        return super().__delitem__(key)

    def __vs_del__(self, core_id: int) -> None:
        self.clear()


class DynamicClipsCache(vs_object, dict[T, VideoNodeT]):
    def __init__(self, cache_size: int = 2) -> None:
        self.cache_size = cache_size

    @abstractmethod
    def get_clip(self, key: T) -> VideoNodeT: ...

    def __getitem__(self, key: T) -> VideoNodeT:
        if key not in self:
            self[key] = self.get_clip(key)

            if len(self) > self.cache_size:
                del self[next(iter(self.keys()))]

        return super().__getitem__(key)


class FramesCache(vs_object, Generic[NodeT, FrameT], dict[int, FrameT]):
    def __init__(self, clip: NodeT, cache_size: int = 10) -> None:
        self.clip = clip
        self.cache_size = cache_size

    def add_frame(self, n: int, f: FrameT) -> FrameT:
        self[n] = f.copy()
        return self[n]

    def get_frame(self, n: int, f: FrameT) -> FrameT:
        return self[n]

    def __setitem__(self, key: int, value: FrameT) -> None:
        super().__setitem__(key, value)

        if len(self) > self.cache_size:
            del self[next(iter(self.keys()))]

    def __getitem__(self, key: int) -> FrameT:
        if key not in self:
            self.add_frame(key, cast(FrameT, self.clip.get_frame(key)))

        return super().__getitem__(key)

    def __vs_del__(self, core_id: int) -> None:
        self.clear()

        if not TYPE_CHECKING:
            self.clip = None


class NodeFramesCache(vs_object, dict[NodeT, FramesCache[NodeT, FrameT]]):
    def _ensure_key(self, key: NodeT) -> None:
        if key not in self:
            super().__setitem__(key, FramesCache(key))

    def __setitem__(self, key: NodeT, value: FramesCache[NodeT, FrameT]) -> None:
        self._ensure_key(key)

        return super().__setitem__(key, value)

    def __getitem__(self, key: NodeT) -> FramesCache[NodeT, FrameT]:
        self._ensure_key(key)

        return super().__getitem__(key)

    def __vs_del__(self, core_id: int) -> None:
        self.clear()


class ClipFramesCache(NodeFramesCache[vs.VideoNode, vs.VideoFrame]): ...


class SceneBasedDynamicCache(DynamicClipsCache[int, vs.VideoNode]):
    def __init__(self, clip: vs.VideoNode, keyframes: Keyframes | str, cache_size: int = 5) -> None:
        super().__init__(cache_size)

        self.clip = clip
        self.keyframes = Keyframes.from_param(clip, keyframes)

    @abstractmethod
    def get_clip(self, key: int) -> vs.VideoNode: ...

    def get_eval(self) -> vs.VideoNode:
        return self.clip.std.FrameEval(lambda n: self[self.keyframes.scenes.indices[n]])

    @classmethod
    def from_clip(cls, clip: vs.VideoNode, keyframes: Keyframes | str, *args: Any, **kwargs: Any) -> vs.VideoNode:
        return cls(clip, keyframes, *args, **kwargs).get_eval()


class NodesPropsCache(vs_object, dict[tuple[NodeT, int], MutableMapping[str, "_VapourSynthMapValue"]]):
    def __delitem__(self, key: tuple[NodeT, int]) -> None:
        if key not in self:
            return

        return super().__delitem__(key)

    def __vs_del__(self, core_id: int) -> None:
        self.clear()


def cache_clip(_clip: NodeT, cache_size: int = 10) -> NodeT:
    if isinstance(_clip, vs.VideoNode):
        cache = FramesCache[vs.VideoNode, vs.VideoFrame](_clip, cache_size)

        blank = vs.core.std.BlankClip(_clip)

        _to_cache_node = vs.core.std.ModifyFrame(blank, _clip, cache.add_frame)
        _from_cache_node = vs.core.std.ModifyFrame(blank, blank, cache.get_frame)

        return cast(NodeT, vs.core.std.FrameEval(blank, lambda n: _from_cache_node if n in cache else _to_cache_node))

    # elif isinstance(_clip, vs.AudioNode):
    #     ...

    return _clip
