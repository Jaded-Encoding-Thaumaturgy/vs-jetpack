# vs-jetpack

[![Documentation](https://img.shields.io/badge/API%20Docs-purple)](https://jaded-encoding-thaumaturgy.github.io/vs-jetpack/)
[![Coverage Status](https://coveralls.io/repos/github/Jaded-Encoding-Thaumaturgy/vs-jetpack/badge.svg?branch=main)](https://coveralls.io/github/Jaded-Encoding-Thaumaturgy/vs-jetpack?branch=main)
[![PyPI Version](https://img.shields.io/pypi/v/vsjetpack)](https://pypi.org/project/vsjetpack/)

`vs-jetpack` provides a collection of Python modules for filtering video using [VapourSynth](https://github.com/vapoursynth/vapoursynth).
These include modules for scaling, masking, denoising, debanding, dehaloing, deinterlacing,
and antialiasing, as well as general utility functions.

For support you can check out the [JET Discord server](https://discord.gg/XTpc6Fa9eB).

## Documentation

You can find the full API reference on the project's documentation [site](https://jaded-encoding-thaumaturgy.github.io/vs-jetpack/api/vstools/).

If you're looking for workflow recommendations, the JET Encoding Guide is available [here](https://github.com/Jaded-Encoding-Thaumaturgy/JET-guide).

## Installation

`vsjetpack` relies on a number of VapourSynth plugins to function.
Most of these plugins are now available as Python packages on PyPI and can be installed automatically using **extras**:

### Breakdown

Most extras are hierarchical. For example, `denoise` includes all plugins from `aa`, which in turn includes `mask`, and so on.

| Extra             | Purpose              | Included Plugins / Packages                                                       |
| :---------------- | :------------------- | :-------------------------------------------------------------------------------- |
| **`source`**      | Clip Indexing        | `bestsource`, `ffms2`, `lsmas`, `d2vsource`, `dvdsrc2`                            |
| **`kernels`**     | Resizing             | `resize2`, `descale`, `vs-placebo`                                                |
| **`rg`**          | Repair & Smoothing   | `awarp`, `zsmooth` (+ `kernels`, `expr`)                                          |
| **`mask`**        | Masking              | `adaptivegrain`, `edgemasks`, `hysteresis`, `subtext` (+ `source`, `rg`)          |
| **`aa`**          | Anti-aliasing        | `bwdif`, `eedi3`, `sangnom`, `sneedif`, `znedi3` (+ `mask`)                       |
| **`denoise`**     | Denoising            | `bm3d`, `dfttest2`, `deblock`, `mvtools`, `nlm-ispc`, `wnnm` (+ `aa`)             |
| **`deband`**      | Debanding            | `vsnoise` (+ `denoise`)                                                           |
| **`deinterlace`** | Deinterlacing        | `dmetrics`, `vivtc` (+ `denoise`)                                                 |
| **`full`**        | All CPU-based extras | All of the above                                                                  |
| **`gpu`**         | Generic GPU          | `ncnn`, `ort`                                                                     |
| **`cl`**          | Open CL              | `knlmeanscl`, `vszipcl`                                                           |
| **`nvidia`**      | NVIDIA GPU           | `bm3dcuda`, `nlm-cuda`, `dfttest2-[nvrtc,cuda]`, `ort-cuda`, `trt{-rtx}` (+ `cl`) |
| **`amd`**         | AMD GPU              | `bm3dhip`, `dfttest2-[hiprtc,hipfft]`, `nlm-hip` (+ `cl`, `gpu`)                  |

> [!IMPORTANT]
> Some plugins distribute their wheels through our custom package index instead of PyPI.
>
> Add `--extra-index-url` to ensure pip can locate all required packages:
>
> ```bash
> pip install vsjetpack[full] --extra-index-url https://jaded-encoding-thaumaturgy.github.io/vs-wheels/simple
> ```
>
> When downloading the `nvidia` extra, you can also add the NVIDIA index:
>
> #### Bash
>
> ```
> pip install vsjetpack[full,nvidia] \
>   --extra-index-url https://pypi.nvidia.com/ \
>   --extra-index-url https://jaded-encoding-thaumaturgy.github.io/vs-wheels/simple
> ```
>
> #### Powershell
>
> ```
> pip install vsjetpack[full,nvidia] `
>   --extra-index-url https://pypi.nvidia.com/ `
>   --extra-index-url https://jaded-encoding-thaumaturgy.github.io/vs-wheels/simple
> ```
>
> For more information, [click here](https://github.com/Jaded-Encoding-Thaumaturgy/vs-wheels).

Not all extras have prebuilt wheels on every platform:

| Extra         | Windows x64 | Linux (glibc 2.35+) | Linux (musl 1.2+) | macOS (Intel/ARM) 15.0+ |
| :------------ | :---------: | :-----------------: | :---------------: | :---------------------: |
| `source`      |     ✅      |         ✅          |        ✅         |           ✅            |
| `kernels`     |     ✅      |         ✅          |        ✅         |           ✅            |
| `rg`          |     ✅      |         ✅          |        ✅         |           ✅            |
| `mask`        |     ✅      |         ✅          |        ✅         |           ✅            |
| `scale`       |     ✅      |         ✅          |        ❌         |           ✅            |
| `aa`          |     ✅      |         ✅          |        ❌         |           ✅            |
| `denoise`     |     ✅      |         ✅          |        ❌         |         ⚠️[^1]          |
| `deband`      |     ✅      |         ✅          |        ❌         |           ✅            |
| `deinterlace` |     ✅      |         ✅          |        ❌         |           ✅            |
| `full`        |     ✅      |         ✅          |        ❌         |           ⚠️            |
| `cl`          |     ✅      |         ✅          |        ❌         |           ✅            |
| `nvidia`      |     ✅      |       ⚠️[^2]        |        ❌         |           ❌            |
| `amd`         |     ✅      |         ✅          |        ❌         |           ❌            |

> [!TIP]
> If a plugin is unavailable for your platform, you may need to build it manually.
>
> Refer to the plugin repository for platform-specific build instructions.

[^1]: `wnnm` isn't available on macOS.

[^2]: Requires Glibc 2.39+ for `bilateralgpu`.
