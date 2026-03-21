from __future__ import annotations

from abc import abstractmethod
from collections import OrderedDict, UserDict
from collections.abc import MutableMapping
from typing import TYPE_CHECKING

from ..vs_proxy import VSObject, VSObjectABC, vs

if TYPE_CHECKING:
    from vapoursynth import _PropValue  # pyright: ignore[reportMissingModuleSource]


__all__ = [
    "ClipFramesCache",
    "ClipsCache",
    "DynamicClipsCache",
    "FramesCache",
    "LRUCache",
    "NodeFramesCache",
    "NodesPropsCache",
    "cache_clip",
]


class ClipsCache(VSObjectABC, UserDict[vs.VideoNode, vs.VideoNode]):
    def __delitem__(self, key: vs.VideoNode) -> None:
        if key not in self:
            return None

        return super().__delitem__(key)


class DynamicClipsCache[T](VSObjectABC, UserDict[T, vs.VideoNode]):
    def __init__(self, cache_size: int = 2) -> None:
        super().__init__()
        self.cache_size = cache_size

    @abstractmethod
    def get_clip(self, key: T) -> vs.VideoNode: ...

    def __getitem__(self, key: T) -> vs.VideoNode:
        if key not in self:
            self[key] = self.get_clip(key)

            if len(self) > self.cache_size:
                del self[next(iter(self.keys()))]

        return super().__getitem__(key)


class LRUCache[K, V](VSObject, OrderedDict[K, V]):
    def __init__(self, cache_size: int = 10) -> None:
        super().__init__()
        self._cache_size = cache_size

    def __getitem__(self, key: K) -> V:
        val = super().__getitem__(key)
        self.move_to_end(key)
        return val

    def __setitem__(self, key: K, value: V) -> None:
        super().__setitem__(key, value)
        self.move_to_end(key)
        self.prune()

    @property
    def cache_size(self) -> int:
        return self._cache_size

    @cache_size.setter
    def cache_size(self, value: int) -> None:
        self._cache_size = value
        self.prune()

    @cache_size.deleter
    def cache_size(self) -> None:
        self._cache_size = 0
        self.prune()

    def prune(self) -> None:
        while len(self) > self._cache_size:
            self.popitem(last=False)


class FramesCache[NodeT: vs.RawNode, FrameT: vs.RawFrame](VSObjectABC, UserDict[int, FrameT]):
    def __init__(self, clip: NodeT, cache_size: int = 10) -> None:
        super().__init__()
        self.clip: NodeT = clip
        self.cache_size = cache_size

    def add_frame(self, n: int, f: FrameT) -> FrameT:
        f = f.copy()
        self[n] = f
        return f

    def get_frame(self, n: int, f: FrameT) -> FrameT:
        return self[n]

    def __setitem__(self, key: int, value: FrameT) -> None:
        super().__setitem__(key, value)

        if len(self) > self.cache_size:
            del self[next(iter(self.keys()))]

    def __getitem__(self, key: int) -> FrameT:
        if key not in self:
            self.add_frame(key, self.clip.get_frame(key))  # type: ignore[arg-type]

        return super().__getitem__(key)


class NodeFramesCache[NodeT: vs.RawNode, FrameT: vs.RawFrame](VSObjectABC, UserDict[NodeT, FramesCache[NodeT, FrameT]]):
    def _ensure_key(self, key: NodeT) -> None:
        if key not in self:
            super().__setitem__(key, FramesCache(key))

    def __setitem__(self, key: NodeT, value: FramesCache[NodeT, FrameT]) -> None:
        self._ensure_key(key)

        return super().__setitem__(key, value)

    def __getitem__(self, key: NodeT) -> FramesCache[NodeT, FrameT]:
        self._ensure_key(key)

        return super().__getitem__(key)


class ClipFramesCache(NodeFramesCache[vs.VideoNode, vs.VideoFrame]): ...


class NodesPropsCache[NodeT: vs.RawNode](LRUCache[tuple[NodeT, int], MutableMapping[str, "_PropValue"]]):
    def __delitem__(self, key: tuple[NodeT, int]) -> None:
        if key not in self:
            return None

        return super().__delitem__(key)


def cache_clip[NodeT: vs.RawNode](_clip: NodeT, cache_size: int = 10) -> NodeT:
    if isinstance(_clip, vs.VideoNode):
        cache = FramesCache[vs.VideoNode, vs.VideoFrame](_clip, cache_size)

        blank = vs.core.std.BlankClip(_clip)

        to_cache_node = vs.core.std.ModifyFrame(blank, _clip, cache.add_frame)
        from_cache_node = vs.core.std.ModifyFrame(blank, blank, cache.get_frame)

        return vs.core.std.FrameEval(blank, lambda n: from_cache_node if n in cache else to_cache_node)  # type: ignore[return-value]

    # elif isinstance(_clip, vs.AudioNode):
    #     ...

    return _clip
