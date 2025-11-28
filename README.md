# vs-jetpack

[![Documentation](https://img.shields.io/badge/API%20Docs-purple)](https://jaded-encoding-thaumaturgy.github.io/vs-jetpack/) [![Coverage Status](https://coveralls.io/repos/github/Jaded-Encoding-Thaumaturgy/vs-jetpack/badge.svg?branch=main)](https://coveralls.io/github/Jaded-Encoding-Thaumaturgy/vs-jetpack?branch=main) [![PyPI Version](https://img.shields.io/pypi/v/vsjetpack)](https://pypi.org/project/vsjetpack/)

Full suite of filters, wrappers, and helper functions for filtering video using VapourSynth

`vs-jetpack` provides a collection of Python modules for filtering video using VapourSynth.
These include modules for scaling, masking, denoising, debanding, dehaloing, deinterlacing,
and antialiasing, as well as general utility functions.

For support you can check out the [JET Discord server](https://discord.gg/XTpc6Fa9eB). <br><br>

## Documentation

You can find the full API reference on the project's documentation [site](https://jaded-encoding-thaumaturgy.github.io/vs-jetpack/).

If you're looking for workflow recommendations, the JET Encoding Guide is available [here](https://github.com/Jaded-Encoding-Thaumaturgy/JET-guide).

## How to install

`vsjetpack` is distributed via **PyPI**, and the latest stable release can be installed using:

```sh
pip install vsjetpack
```

As of version **1.0.0**, prebuilt wheels are also provided in the [**GitHub Releases**](https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack/releases).
<br><br>

## Dependencies

Note that `vsjetpack` only provides Python functions, many of them wrapping or combining existing plugins.
You will need to install these plugins separately, for example using [vsrepo](https://github.com/vapoursynth/vsrepo).

| **Essential**                                                                          | **Source filters**                                                    | **Optional**                                                                   |                                                                              |
| -------------------------------------------------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| [akarin](https://github.com/Jaded-Encoding-Thaumaturgy/akarin-vapoursynth-plugin) [^1] | [bestsource](https://github.com/vapoursynth/bestsource)               | [adaptivegrain](https://github.com/Irrational-Encoding-Wizardry/adaptivegrain) | [mvtools](https://github.com/dubhater/vapoursynth-mvtools)                   |
| [fmtconv](https://gitlab.com/EleonoreMizo/fmtconv/) [^1]                               | [carefulsource](https://github.com/wwww-wwww/carefulsource)           | [awarp](https://github.com/HolyWu/VapourSynth-AWarp)                           | [neo_f3kdb](https://github.com/HomeOfAviSynthPlusEvolution/neo_f3kdb)        |
| [resize2](https://github.com/Jaded-Encoding-Thaumaturgy/vapoursynth-resize2) [^1]      | [d2vsource](https://github.com/dwbuiten/d2vsource)                    | [bilateralgpu](https://github.com/WolframRhodium/VapourSynth-BilateralGPU)     | [nlm-cuda](https://github.com/AmusementClub/vs-nlm-cuda)                     |
| [vszip](https://github.com/dnjulek/vapoursynth-zip)                                    | [dvdsrc2](https://github.com/jsaowji/dvdsrc2)                         | [bm3d](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-BM3D)         | [nlm-ispc](https://github.com/AmusementClub/vs-nlm-ispc)                     |
| [zsmooth](https://github.com/adworacz/zsmooth)                                         | [ffms2](https://github.com/FFMS/ffms2)                                | [bm3dcuda](https://github.com/WolframRhodium/VapourSynth-BM3DCUDA)             | [placebo](https://github.com/sgt0/vs-placebo)                                |
|                                                                                        | [imwri](https://github.com/vapoursynth/vs-imwri)                      | [bwdif](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-Bwdif)       | [sangnom](https://github.com/dubhater/vapoursynth-sangnom)                   |
|                                                                                        | [lsmas](https://github.com/HomeOfAviSynthPlusEvolution/L-SMASH-Works) | [dctfilter](https://github.com/AmusementClub/VapourSynth-DCTFilter)            | [scxvid](https://github.com/dubhater/vapoursynth-scxvid)                     |
|                                                                                        |                                                                       | [deblock](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-Deblock)   | [sneedif](https://github.com/Jaded-Encoding-Thaumaturgy/vapoursynth-SNEEDIF) |
|                                                                                        |                                                                       | [descale](https://github.com/Jaded-Encoding-Thaumaturgy/vapoursynth-descale)   | [tcanny](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-TCanny)   |
|                                                                                        |                                                                       | [dfttest](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-DFTTest)   | [tedgemask](https://github.com/dubhater/vapoursynth-tedgemask)               |
|                                                                                        |                                                                       | [dfttest2](https://github.com/AmusementClub/vs-dfttest2)                       | [vivtc](https://github.com/vapoursynth/vivtc)                                |
|                                                                                        |                                                                       | [edgemasks](https://github.com/HolyWu/VapourSynth-EdgeMasks)                   | [vs-mlrt](https://github.com/AmusementClub/vs-mlrt)                          |
|                                                                                        |                                                                       | [eedi2](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-EEDI2)       | [vs-noise](https://github.com/wwww-wwww/vs-noise)                            |
|                                                                                        |                                                                       | [eedi3](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-EEDI3)       | [wnnm](https://github.com/AmusementClub/VapourSynth-WNNM)                    |
|                                                                                        |                                                                       | [hysteresis](https://github.com/sgt0/vapoursynth-hysteresis)                   | [wwxd](https://github.com/dubhater/vapoursynth-wwxd)                         |
|                                                                                        |                                                                       | [knlmeanscl](https://github.com/Khanattila/KNLMeansCL)                         | [znedi3](https://github.com/sekrit-twc/znedi3)                               |
|                                                                                        |                                                                       | [manipmv](https://github.com/Mikewando/manipulate-motion-vectors)              |                                                                              |

[^1]: Can be considered mandatory
