from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, MutableMapping

from ..vs_proxy import vs, vs_object

if TYPE_CHECKING:
    from vapoursynth import _PropValue  # pyright: ignore[reportMissingModuleSource]


__all__ = [
    "ClipFramesCache",
    "ClipsCache",
    "DynamicClipsCache",
    "FramesCache",
    "NodeFramesCache",
    "NodesPropsCache",
    "cache_clip",
]


class ClipsCache(vs_object, dict[vs.VideoNode, vs.VideoNode]):
    def __delitem__(self, key: vs.VideoNode) -> None:
        if key not in self:
            return

        return super().__delitem__(key)

    def __vs_del__(self, core_id: int) -> None:
        self.clear()


class DynamicClipsCache[T](vs_object, dict[T, vs.VideoNode]):
    def __init__(self, cache_size: int = 2) -> None:
        self.cache_size = cache_size

    @abstractmethod
    def get_clip(self, key: T) -> vs.VideoNode: ...

    def __getitem__(self, key: T) -> vs.VideoNode:
        if key not in self:
            self[key] = self.get_clip(key)

            if len(self) > self.cache_size:
                del self[next(iter(self.keys()))]

        return super().__getitem__(key)

    def __vs_del__(self, core_id: int) -> None:
        self.clear()


class FramesCache[_NodeT: vs.RawNode, _FrameT: vs.RawFrame](vs_object, dict[int, _FrameT]):
    def __init__(self, clip: _NodeT, cache_size: int = 10) -> None:
        self.clip: _NodeT = clip
        self.cache_size = cache_size

    def add_frame(self, n: int, f: _FrameT) -> _FrameT:
        f = f.copy()
        self[n] = f
        return f

    def get_frame(self, n: int, f: _FrameT) -> _FrameT:
        return self[n]

    def __setitem__(self, key: int, value: _FrameT) -> None:
        super().__setitem__(key, value)

        if len(self) > self.cache_size:
            del self[next(iter(self.keys()))]

    def __getitem__(self, key: int) -> _FrameT:
        if key not in self:
            self.add_frame(key, self.clip.get_frame(key))  # type: ignore[arg-type]

        return super().__getitem__(key)

    def __vs_del__(self, core_id: int) -> None:
        self.clear()
        del self.clip


class NodeFramesCache[_NodeT: vs.RawNode, _FrameT: vs.RawFrame](vs_object, dict[_NodeT, FramesCache[_NodeT, _FrameT]]):
    def _ensure_key(self, key: _NodeT) -> None:
        if key not in self:
            super().__setitem__(key, FramesCache(key))

    def __setitem__(self, key: _NodeT, value: FramesCache[_NodeT, _FrameT]) -> None:
        self._ensure_key(key)

        return super().__setitem__(key, value)

    def __getitem__(self, key: _NodeT) -> FramesCache[_NodeT, _FrameT]:
        self._ensure_key(key)

        return super().__getitem__(key)

    def __vs_del__(self, core_id: int) -> None:
        self.clear()


class ClipFramesCache(NodeFramesCache[vs.VideoNode, vs.VideoFrame]): ...


class NodesPropsCache[_NodeT: vs.RawNode](vs_object, dict[tuple[_NodeT, int], MutableMapping[str, "_PropValue"]]):
    def __delitem__(self, key: tuple[_NodeT, int]) -> None:
        if key not in self:
            return

        return super().__delitem__(key)

    # def __vs_del__(self, core_id: int) -> None:
    #     self.clear()


def cache_clip[_NodeT: vs.RawNode](_clip: _NodeT, cache_size: int = 10) -> _NodeT:
    if isinstance(_clip, vs.VideoNode):
        cache = FramesCache[vs.VideoNode, vs.VideoFrame](_clip, cache_size)

        blank = vs.core.std.BlankClip(_clip)

        _to_cache_node = vs.core.std.ModifyFrame(blank, _clip, cache.add_frame)
        _from_cache_node = vs.core.std.ModifyFrame(blank, blank, cache.get_frame)

        return vs.core.std.FrameEval(blank, lambda n: _from_cache_node if n in cache else _to_cache_node)  # type: ignore[return-value]

    # elif isinstance(_clip, vs.AudioNode):
    #     ...

    return _clip
