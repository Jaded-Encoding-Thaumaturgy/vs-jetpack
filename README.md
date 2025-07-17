# vs-jetpack

[![Coverage Status](https://coveralls.io/repos/github/Jaded-Encoding-Thaumaturgy/vs-jetpack/badge.svg?branch=main)](https://coveralls.io/github/Jaded-Encoding-Thaumaturgy/vs-jetpack?branch=main)
[![Documentation](https://img.shields.io/badge/API%20Docs-purple)](https://jaded-encoding-thaumaturgy.github.io/vs-jetpack/)



Full suite of filters, wrappers, and helper functions for filtering video using VapourSynth

`vs-jetpack` provides a collection of Python modules for filtering video using VapourSynth.
These include modules for scaling, masking, denoising, debanding, dehaloing, deinterlacing,
and antialiasing, as well as general utility functions.

For support you can check out the [JET Discord server](https://discord.gg/XTpc6Fa9eB). <br><br>

## How to install

Install `vsjetpack` with the following command:

```sh
pip install vsjetpack
```

Or if you want the latest git version, install it with this command:

```sh
pip install git+https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack.git
```

Note that `vsjetpack` only provides Python functions,
many of them wrapping or combining existing plugins.
You will need to install these plugins separately,
for example using [vsrepo](https://github.com/vapoursynth/vsrepo).

#### Dependencies

| **Essential**                          | **Source filters**                     | **Optional**                          |
|----------------------------------------|----------------------------------------|---------------------------------------|
| [akarin](https://github.com/AkarinVS/vapoursynth-plugin) [^1] | [bestsource](https://github.com/vapoursynth/bestsource) | [eedi2](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-EEDI2) / [eedi2cuda](https://github.com/hooke007/VapourSynth-EEDI2CUDA) |
| [resize2](https://github.com/Jaded-Encoding-Thaumaturgy/vapoursynth-resize2) [^1] | [d2vsource](https://github.com/dwbuiten/d2vsource) | [eedi3](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-EEDI3) |
| [vszip](https://github.com/dnjulek/vapoursynth-zip) | [dvdsrc2](https://github.com/jsaowji/dvdsrc2) | [znedi3](https://github.com/sekrit-twc/znedi3) / [sneedif](https://github.com/Jaded-Encoding-Thaumaturgy/vapoursynth-SNEEDIF) |
| [fmtconv](https://gitlab.com/EleonoreMizo/fmtconv/) | [ffms2](https://github.com/FFMS/ffms2) | [sangnom](https://github.com/dubhater/vapoursynth-sangnom) |
|                                        | [lsmas](https://github.com/HomeOfAviSynthPlusEvolution/L-SMASH-Works) | [neo_f3kdb](https://github.com/HomeOfAviSynthPlusEvolution/neo_f3kdb) |
|                                        | [imwri](https://github.com/vapoursynth/vs-imwri) | [chickendream](https://gitlab.com/EleonoreMizo/chickendream/) / [fgrain_cuda](https://github.com/AmusementClub/vs-fgrain-cuda) |
|                                        | [carefulsource](https://github.com/wwww-wwww/carefulsource) | [vs-noise](https://github.com/wwww-wwww/vs-noise) |
|                                        |                                        | [vivtc](https://github.com/vapoursynth/vivtc) |
|                                        |                                        | [wnnm](https://github.com/WolframRhodium/VapourSynth-WNNM) |
|                                        |                                        | [bm3d](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-BM3D) / [bm3dcuda](https://github.com/WolframRhodium/VapourSynth-BM3DCUDA) |
|                                        |                                        | [dctfilter](https://github.com/AmusementClub/VapourSynth-DCTFilter) |
|                                        |                                        | [mlrt](https://github.com/AmusementClub/vs-mlrt) |
|                                        |                                        | [dfttest2](https://github.com/AmusementClub/vs-dfttest2) / [dfttest](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-DFTTest) |
|                                        |                                        | [mvtools](https://github.com/dubhater/vapoursynth-mvtools) / [mvtools-sf](https://github.com/IFeelBloated/vapoursynth-mvtools-sf) |
|                                        |                                        | [manipmv](https://github.com/Mikewando/manipulate-motion-vectors) |
|                                        |                                        | [scxvid](https://github.com/dubhater/vapoursynth-scxvid) / [wwxd](https://github.com/dubhater/vapoursynth-wwxd) |
|                                        |                                        | [bwdif](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-Bwdif) |
|                                        |                                        | [fft3dfilter](https://github.com/AmusementClub/VapourSynth-FFT3DFilter) |
|                                        |                                        | [nlm-ispc](https://github.com/AmusementClub/vs-nlm-ispc) / [nlm-cuda](https://github.com/AmusementClub/vs-nlm-cuda) / [knlmeanscl](https://github.com/Khanattila/KNLMeansCL) |
|                                        |                                        | [descale](https://github.com/Jaded-Encoding-Thaumaturgy/vapoursynth-descale) |
|                                        |                                        | [placebo](https://github.com/sgt0/vs-placebo) |
|                                        |                                        | [awarpsharp2](https://github.com/dubhater/vapoursynth-awarpsharp2) / [warpsharpsf](https://github.com/IFeelBloated/warpsharp) |
|                                        |                                        | [tcanny](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-TCanny) |
|                                        |                                        | [tedgemask](https://github.com/dubhater/vapoursynth-tedgemask) |
|                                        |                                        | [hysteresis](https://github.com/sgt0/vapoursynth-hysteresis) |
|                                        |                                        | [adaptivegrain](https://github.com/Irrational-Encoding-Wizardry/adaptivegrain) |
|                                        |                                        | [zsmooth](https://github.com/adworacz/zsmooth) |
|                                        |                                        | [bilateralgpu](https://github.com/WolframRhodium/VapourSynth-BilateralGPU) |

[^1]: Can be considered mandatory
