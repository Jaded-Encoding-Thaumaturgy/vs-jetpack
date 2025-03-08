from typing import NamedTuple


__all__ = [
    # Versioning
    '__version__', '__api_version__', 'PluginVersion',
]

###
# VapourSynth Versioning

class VapourSynthVersion(NamedTuple):
    release_major: int
    release_minor: int


class VapourSynthAPIVersion(NamedTuple):
    api_major: int
    api_minor: int


__version__: VapourSynthVersion
__api_version__: VapourSynthAPIVersion


###
# Plugin Versioning


class PluginVersion(NamedTuple):
    major: int
    minor: int