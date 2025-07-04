from __future__ import annotations

from abc import abstractmethod
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Generic, MutableMapping, TypeVar, cast

from jetpytools import T

from ..functions import Keyframes
from ..types import vs_object, VideoNodeT
from . import vs_proxy as vs

if TYPE_CHECKING:
    from vapoursynth._typings import _VapourSynthMapValue
else:
    _VapourSynthMapValue = Any


__all__ = [
    'ClipsCache',

    'DynamicClipsCache',

    'FramesCache',

    'NodeFramesCache',

    'ClipFramesCache',

    'SceneBasedDynamicCache',

    'NodesPropsCache',

    'cache_clip',

    'GlobalCache',

    'get_cached_value',

    'set_cached_value',

    'clear_cache',

    'cache_value'
]


NodeT = TypeVar('NodeT', bound=vs.RawNode)
FrameT = TypeVar('FrameT', bound=vs.RawFrame)
CacheKeyT = TypeVar('CacheKeyT')
CacheValueT = TypeVar('CacheValueT')


class GlobalCache:
    """Thread-safe global cache for storing arbitrary values."""

    def __init__(self) -> None:
        self._cache: dict[Any, Any] = {}
        self._lock = Lock()

    def get(self, key: Any, default: Any = None) -> Any:
        """
        Get a value from cache, and return a default value if not found.

        Args:
            key: Cache key
            default: Default value to return if key not found

        Returns:
            The cached value or the default value if not found
        """

        with self._lock:
            return self._cache.get(key, default)

    def set(self, key: Any, value: Any) -> None:
        """
        Set a value in cache.

        If the key already exists, the value will be updated.

        Args:
            key: Cache key
            value: Value to cache
        """

        with self._lock:
            self._cache[key] = value

    def delete(self, key: Any) -> bool:
        """
        Delete a key from cache, and return True if key existed.

        Args:
            key: Cache key

        Returns:
            True if key existed, False otherwise
        """

        with self._lock:
            if key in self._cache:
                del self._cache[key]

                return True

            return False

    def clear(self) -> None:
        """Clear all cache entries."""

        with self._lock:
            self._cache.clear()

    def get_or_set(self, key: Any, callback: Callable[[], Any], force: bool = False) -> Any:
        """
        Get a value from cache, or compute it using callback if not found.

        Args:
            key: Cache key
            callback: Function to call if value needs to be computed
            force: If True, always recompute the value

        Returns:
            The cached or computed value
        """

        if not force:
            if (cached_value := self.get(key)) is not None:
                return cached_value

        new_value = callback()
        self.set(key, new_value)

        return new_value

    def __len__(self) -> int:
        """
        Return number of cache entries.

        Returns:
            The number of cache entries
        """

        with self._lock:
            return len(self._cache)

    def __contains__(self, key: Any) -> bool:
        """
        Check if key exists.

        Args:
            key: Cache key

        Returns:
            True if key exists, False otherwise
        """

        with self._lock:
            return key in self._cache


_global_cache = GlobalCache()


def get_cached_value(key: Any, default: Any = None) -> Any:
    """
    Get a value from the global cache.

    This function should be used if a value if expected to be cached already.
    If you're uncertain, use `func:cache_value` instead.

    Args:
        key: Cache key
        default: Default value to return if key not found

    Returns:
        The cached value or the default value if not found
    """

    return _global_cache.get(key, default)


def set_cached_value(key: Any, value: Any) -> None:
    """
    Set a value in the global cache.

    Args:
        key: Cache key
        value: Value to cache
    """

    _global_cache.set(key, value)


def clear_cache() -> None:
    """Clear the global cache."""

    _global_cache.clear()


def cache_value(key: Any, callback: Callable[[], Any], force: bool = False) -> Any:
    """
    Get a value from cache or compute it using callback.

    This is the main function for caching expensive operations like plugin version checks,
    optimization parameter calculations, etc.

    Example:

        .. code-block:: python
            # Cache vs-jetpack version check
            >>> from importlib.metadata import version as fetch_version
            >>> from packaging.version import Version
            >>> version = cache_value('vsjetpack_version', lambda: Version(fetch_version('vsjetpack')))
            >>> version
            ... <Version('0.4.0')>
            >>> cache_value('vsjetpack_version')
            ... <Version('0.4.0')>

    Args:
        key: Unique identifier for the cached value
        callback: Function that returns the value to cache
        force: If True, always recompute the value

    Returns:
        The cached or computed value
    """

    return _global_cache.get_or_set(key, callback, force)


class ClipsCache(vs_object, dict[vs.VideoNode, vs.VideoNode]):
    def __delitem__(self, __key: vs.VideoNode) -> None:
        if __key not in self:
            return

        return super().__delitem__(__key)

    def __vs_del__(self, core_id: int) -> None:
        self.clear()


class DynamicClipsCache(vs_object, dict[T, VideoNodeT]):
    def __init__(self, cache_size: int = 2) -> None:
        self.cache_size = cache_size

    @abstractmethod
    def get_clip(self, key: T) -> VideoNodeT:
        ...

    def __getitem__(self, __key: T) -> VideoNodeT:
        if __key not in self:
            self[__key] = self.get_clip(__key)

            if len(self) > self.cache_size:
                del self[next(iter(self.keys()))]

        return super().__getitem__(__key)


class FramesCache(vs_object, Generic[NodeT, FrameT], dict[int, FrameT]):
    def __init__(self, clip: NodeT, cache_size: int = 10) -> None:
        self.clip = clip
        self.cache_size = cache_size

    def add_frame(self, n: int, f: FrameT) -> FrameT:
        self[n] = f.copy()
        return self[n]

    def get_frame(self, n: int, f: FrameT) -> FrameT:
        return self[n]

    def __setitem__(self, __key: int, __value: FrameT) -> None:
        super().__setitem__(__key, __value)

        if len(self) > self.cache_size:
            del self[next(iter(self.keys()))]

    def __getitem__(self, __key: int) -> FrameT:
        if __key not in self:
            self.add_frame(__key, cast(FrameT, self.clip.get_frame(__key)))

        return super().__getitem__(__key)

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


class ClipFramesCache(NodeFramesCache[vs.VideoNode, vs.VideoFrame]):
    ...


class SceneBasedDynamicCache(DynamicClipsCache[int, vs.VideoNode]):
    def __init__(self, clip: vs.VideoNode, keyframes: Keyframes | str, cache_size: int = 5) -> None:
        super().__init__(cache_size)

        self.clip = clip
        self.keyframes = Keyframes.from_param(clip, keyframes)

    @abstractmethod
    def get_clip(self, key: int) -> vs.VideoNode:
        ...

    def get_eval(self) -> vs.VideoNode:
        return self.clip.std.FrameEval(lambda n: self[self.keyframes.scenes.indices[n]])

    @classmethod
    def from_clip(cls, clip: vs.VideoNode, keyframes: Keyframes | str, *args: Any, **kwargs: Any) -> vs.VideoNode:
        return cls(clip, keyframes, *args, **kwargs).get_eval()


class NodesPropsCache(vs_object, dict[tuple[NodeT, int], MutableMapping[str, _VapourSynthMapValue]]):
    def __delitem__(self, __key: tuple[NodeT, int]) -> None:
        if __key not in self:
            return

        return super().__delitem__(__key)

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
