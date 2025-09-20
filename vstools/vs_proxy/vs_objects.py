from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any, Callable

from typing_extensions import Self

from .vs_proxy import core, register_on_creation

__all__ = ["vs_object"]


class vs_object:  # noqa: N801
    """
    Special object that follows the lifecycle of the VapourSynth environment/core.

    If a special dunder is created, __vs_del__, it will be called when the environment is getting deleted.

    This is especially useful if you have to hold a reference to a VideoNode or Plugin/Function object
    in this object as you need to remove it for the VapourSynth core to be freed correctly.
    """

    __vsdel_partial_register: Callable[..., None]
    __vsdel_register: Callable[[int], None] | None = None

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        try:
            self = super().__new__(cls, *args, **kwargs)
        except TypeError:
            self = super().__new__(cls)

        if hasattr(self, "__vs_del__"):

            def _register(core_id: int) -> None:
                def __vsdel_partial_register(core_id: int) -> None:
                    self.__vs_del__(core_id)

                self.__vsdel_partial_register = partial(__vsdel_partial_register, core_id)

                core.register_on_destroy(self.__vsdel_partial_register)

            # [un]register_on_creation/destroy will only hold a weakref to the object
            self.__vsdel_register = _register
            register_on_creation(self.__vsdel_register)

        return self

    if TYPE_CHECKING:

        def __vs_del__(self, core_id: int) -> None:
            """
            Special dunder that will be called when a core is getting freed.
            """
