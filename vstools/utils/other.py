from subprocess import run

__all__ = ["get_nvidia_version", "is_gpu_available"]


def _str_to_ver(string: str) -> tuple[int, int]:
    return tuple(int(x) for x in string.strip().split(".", 2))  # type: ignore


def get_nvidia_version() -> tuple[int, int] | None:
    """
    Check if nvidia drivers are installed and if available return the version.
    """

    try:
        smi = run(["nvidia-smi", "-q"], capture_output=True)
    except FileNotFoundError:
        pass
    else:
        if not smi.returncode:
            return _str_to_ver(smi.stdout.splitlines()[5].decode().split(":")[-1])

    return None


def is_gpu_available() -> bool:
    """
    Check if any GPU is available.
    """

    try:
        smi = run(["nvidia-smi"], capture_output=True, text=True)
    except FileNotFoundError:
        pass
    else:
        if smi.returncode == 0:
            return True

    try:
        rocm_smi = run(["rocm-smi"], capture_output=True, text=True)
    except FileNotFoundError:
        pass
    else:
        if rocm_smi.returncode == 0:
            return True

    return False
