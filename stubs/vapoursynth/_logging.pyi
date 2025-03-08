from typing import NoReturn


__all__ = ['LogHandle', 'Error']


class LogHandle:
    def __init__(self) -> NoReturn: ...


class Error(Exception): ...
