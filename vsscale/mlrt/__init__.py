"""
# VapourSynth Machine Learning RunTime (MLRT)

MLRT provides a unified interface for executing machine learning models
via various VapourSynth backend plugins ([VS-MLRT](https://github.com/AmusementClub/vs-mlrt)),
alongside a command-line interface (CLI) for downloading and managing model assets and compiled engine artifacts.

## Command Line Interface (CLI)

The CLI tool `vsscale` provides commands to manage models (ONNX) and built engines/caches (artifacts).

#### Commands:
- **Download ONNX models**:
  ```bash
  # Interactive mode (prompts for provider, version/tag, and assets)
  vsscale onnx download

  # Download a specific provider (e.g., ArtCNN)
  vsscale onnx download ArtCNN

  # Download a specific version of a provider
  vsscale onnx download ArtCNN==v1.6.2

  # Download the latest release of a provider automatically
  vsscale onnx download ArtCNN --latest
  ```
- **List files**:
  ```bash
  # List downloaded ONNX models
  vsscale onnx show

  # List compiled TensorRT/MIGraphX engines/caches
  vsscale artifact show
  ```
- **Clear files**:
  ```bash
  # Delete downloaded ONNX models
  vsscale onnx clear

  # Delete compiled TensorRT/MIGraphX engines/caches
  vsscale artifact clear
  ```

#### Global VS Local Cache:
By default, files are managed locally within the package storage `.vsjet` folder.

Add the `--global` flag to target the platform-specific user cache directory
(e.g., `AppData\\Local\\vsjet\\vsscale\\Cache` on Windows).

---

## Configuration (TOML & Environment Variables)

Default values for the CLI and library search paths can be configured via files or environment variables.

#### TOML Configuration:
The library parses configurations from `vsjet.toml` or `pyproject.toml` in the working directory.

- **vsjet.toml**:
  ```toml
  [vsscale]
  global = true       # Use the global cache folder by default
  ```

- **pyproject.toml**:
  ```toml
  [tool.vsscale]
  global = true

  [tool.vsscale.onnx.download]
  # This tells the CLI to automatically download the latest release
  # of each model when using `vsscale onnx download` without any arguments.
  provider = ["ArtCNN", "DPIR", "Waifu2x"]
  latest = true
  ```

- When using the ArtCNN, Waifu2x or DPIR classes,
  you can automatically download the models if they're not downloaded yet
  by adding the `auto = true` flag in the `download` section:

  **pyproject.toml**:
  ```toml
  [tool.vsscale.onnx.download]
  provider = ["ArtCNN==v1.6.2", "DPIR==20210902", "Waifu2x==20250502-2"]
  auto = true
  ```


#### Environment Variables:
- `VSSCALE_GLOBAL` / `VSSCALE_ONNX_GLOBAL` / `VSSCALE_ARTIFACT_GLOBAL`:
  Set to `true` to force global storage.
- `VSSCALE_LATEST` / `VSSCALE_ONNX_DOWNLOAD_LATEST`:
  Set to `true` to default to downloading latest releases.

---

## Calling Backends in Python

The package exposes the `Backend` class, containing unified wrappers for several runtime plugins.

#### Automated Backend Selection:

You can automatically select the most suitable backend for your system:
```python
from vsscale import Backend

# Automatically selects the best backend for GPU device 0
backend = Backend.autoselect(device_id=0)
```

#### Manual Backend Selection:

You can explicitly instantiate specific backends and configure their execution details:

- **TensorRT**: `Backend.TRT()`
- **TensorRT RTX**: `Backend.TRT_RTX()`
- **ONNX Runtime (CPU/CUDA/DirectML/CoreML)**: `Backend.ORT_CPU()`, `Backend.ORT_CUDA()`, `Backend.ORT_DML()`,
  `Backend.ORT_CoreML()`
- **OpenVINO (CPU/GPU/NPU)**: `Backend.OV_CPU()`, `Backend.OV_GPU()`, `Backend.OV_NPU()`
- **NCNN (Vulkan)**: `Backend.NCNN()`

#### Running Inference:

- Invoke `inference()` on the backend instance, specifying the clip(s), model path, tile size, and overlap:

  ```python
  from vsscale import Backend

  upscaled = Backend.TRT().inference(
      clip,
      ".vsjet/vsscale/artcnn/v1.6.2/ArtCNN_R8F64.onnx",
      overlap=(0, 0),
      tilesize=(1920, 1080),
  )
  ```

- An easy-to-use wrapper is also available and is the recommended way to use the backend plugins:

  ```python
  from vsscale import ArtCNN, Backend
  from vssource import BestSource
  from vstools import core, depth, get_y, vs
  from vsview import set_output

  clip = BestSource.source("input.mkv", bits=0)
  clip_y = get_y(clip)
  clip_y = depth(clip_y, 16, sample_type=vs.SampleType.FLOAT)

  upscaled = ArtCNN.R8F64(Backend.TRT).supersample(clip_y, rfactor=2)
  set_output(upscaled)
  ```
"""

from .backend import Backend
from .settings import (
    get_artifacts_folder,
    get_global_cache,
    get_local_cache,
    get_model_folder,
    get_provider_folder,
)

__all__ = [
    "Backend",
    "get_artifacts_folder",
    "get_global_cache",
    "get_local_cache",
    "get_model_folder",
    "get_provider_folder",
]
