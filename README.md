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

`vsjetpack` relies on several VapourSynth plugins to function.
Most of these plugins are now available as Python packages on PyPI and can be installed automatically using **extras**:

### Breakdown

Most extras are hierarchical. For example, `denoise` includes all plugins from `aa`, which in turn includes `mask`, and so on.

| Extra             | Purpose              | Included Plugins / Packages                                                          |
| :---------------- | :------------------- | :----------------------------------------------------------------------------------- |
| **`basic`**       | Essential plugins    | `akarin`, `fmtconv`, `vszip`, `bestsource`, `scxvid`, `scipy`, `rich`, `psutil`      |
| **`source`**      | Clip Indexing        | `ffms2`, `d2vsource`, `dvdsrc2`                                                      |
| **`kernels`**     | Resizing             | `resize2`, `descale`, `vs-placebo`                                                   |
| **`rg`**          | Repair & Smoothing   | `awarp`, `zsmooth` (+ `kernels`, `expr`)                                             |
| **`mask`**        | Masking              | `adaptivegrain`, `edgemasks`, `hysteresis`, `subtext` (+ `source`, `rg`)             |
| **`aa`**          | Anti-aliasing        | `bwdif`, `eedi3`, `sangnom`, `sneedif`, `znedi3` (+ `mask`)                          |
| **`denoise`**     | Denoising            | `bm3d`, `dctfilter`, `dfttest2`, `deblock`, `mvtools`, `nlm-ispc`, `wnnm` (+ `aa`)   |
| **`deband`**      | Debanding            | `vsnoise` (+ `denoise`)                                                              |
| **`deinterlace`** | Deinterlacing        | `dmetrics`, `vivtc` (+ `denoise`)                                                    |
| **`nvidia`**      | NVIDIA GPU (CUDA)    | `bm3dcuda`, `bilateralgpu`, `nlm-cuda`, `dfttest2-[nvrtc,cuda]`                      |
| **`amd`**         | AMD/Generic GPU      | `bm3dhip`, `knlmeanscl`, `dfttest2-[hiprtc,hipfft]`                                  |
| **`full`**        | All CPU-based extras | `basic`, `source`, `kernels`, `rg`, `mask`, `aa`, `denoise`, `deband`, `deinterlace` |
| **`full-nvidia`** | Full + NVIDIA GPU    | `full`, `nvidia`                                                                     |
| **`full-amd`**    | Full + AMD GPU       | `full`, `amd`                                                                        |

> [!IMPORTANT]
> Some plugins distribute their wheels through our custom package index instead of PyPI.
>
> Add `--extra-index-url` to ensure pip can locate all required packages:
>
> ```bash
> pip install vsjetpack[basic,source,denoise] --extra-index-url https://jaded-encoding-thaumaturgy.github.io/vs-wheels/simple
> ```
>
> For more information, [click here](https://github.com/Jaded-Encoding-Thaumaturgy/vs-wheels).

Not all extras have prebuilt wheels on every platform:

| Extra        | Windows x64 | Linux (glibc) | Linux (musl) | macOS (Intel/ARM) |
| :----------- | :---------: | :-----------: | :----------: | :---------------: |
| `basic` [^1] |     ✅      |      ✅       |      ✅      |        ✅         |
| `source`     |     ✅      |      ✅       |      ✅      |        ✅         |
| `kernels`    |     ✅      |      ✅       |      ✅      |        ✅         |
| `rg`         |     ✅      |      ✅       |      ✅      |        ✅         |
| `mask`       |     ✅      |      ✅       |      ✅      |        ✅         |
| `aa`         |     ✅      |      ✅       |      ✅      |        ✅         |
| `denoise`    |     ✅      |    ⚠️ [^2]    |      ❌      |      ⚠️ [^4]      |
| `cl`         |     ✅      |      ✅       |      ❌      |        ✅         |
| `nvidia`     |     ✅      |    ⚠️ [^3]    |      ❌      |        ❌         |
| `amd`        |     ✅      |      ✅       |      ❌      |        ❌         |
| `full`       |     ✅      |    ⚠️ [^2]    |      ❌      |        ❌         |

[^1]: Will be merged into the hard requirements in the next version.

[^2]: Requires Glibc 2.34+ for `nlm-ispc`.

[^3]: Requires Glibc 2.39+ for `bilateralgpu`.

[^4]: `wnnm` isn't available on macOS.
