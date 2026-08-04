"""
Drop-in replacement for ``import vapoursynth`` that re-exports the upstream API
and layers additional functionality on top.

Key additions:

- Lazy / proxied core access

    Allows holding references like ``core.std.BlankClip`` without triggering core creation
    or locking into an environment policy.

    * ``core.proxied``: A weakref-backed proxy to the current active core.
    * ``core.lazy``: A completely lazy proxy resolving only on function invocation.

- Core lifecycle hooks

    ``register_on_creation`` / ``unregister_on_creation`` react to core instantiation.

- Policy and environment introspection

    Retrieve environment policies via GC graph walking and check active environment status.

- Interactive compatibility

    Stubs ``__file__`` and the ``__vapoursynth__`` module for Jupyter notebooks, REPLs,
    and IDE interactive windows so VSScript environment policies resolve correctly.

- Extended preset video formats

    Exports missing bit-depth and subsampling variant constants.
"""

from .objects import *
from .vs_vars import *
