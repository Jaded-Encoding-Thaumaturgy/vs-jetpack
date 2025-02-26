# vs-jetpack

[![Coverage Status](https://coveralls.io/repos/github/Jaded-Encoding-Thaumaturgy/vs-jetpack/badge.svg?branch=main)](https://coveralls.io/github/Jaded-Encoding-Thaumaturgy/vs-jetpack?branch=main)


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

Note: Not all plugin dependencies are hard dependencies, many are optional.

- [eedi2](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-EEDI2)
- [eedi3](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-EEDI3)
- [znedi3](https://github.com/sekrit-twc/znedi3) / [sneedif](https://github.com/Jaded-Encoding-Thaumaturgy/vapoursynth-SNEEDIF)
- [sangnom](https://github.com/dubhater/vapoursynth-sangnom)
- [neo_f3kdb](https://github.com/HomeOfAviSynthPlusEvolution/neo_f3kdb)
- [chickendream](https://gitlab.com/EleonoreMizo/chickendream/) / [fgrain_cuda](https://github.com/AmusementClub/vs-fgrain-cuda)
- [vs-noise](https://github.com/wwww-wwww/vs-noise)
- [vivtc](https://github.com/vapoursynth/vivtc)
- [wnnm](https://github.com/WolframRhodium/VapourSynth-WNNM)
- [bm3d](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-BM3D) / [bm3dcuda](https://github.com/WolframRhodium/VapourSynth-BM3DCUDA)
- [bmdegrain](https://github.com/AmusementClub/vs-bmdegrain)
- [dctfilter](https://github.com/AmusementClub/VapourSynth-DCTFilter)
- [mlrt](https://github.com/AmusementClub/vs-mlrt)
- [dfttest2](https://github.com/AmusementClub/vs-dfttest2) / [dfttest](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-DFTTest)
- [mvtools](https://github.com/dubhater/vapoursynth-mvtools) / [mvtools-sf](https://github.com/IFeelBloated/vapoursynth-mvtools-sf)
- [manipmv](https://github.com/Mikewando/manipulate-motion-vectors)
- [bwdif](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-Bwdif)
- [fft3dfilter](https://github.com/AmusementClub/VapourSynth-FFT3DFilter)
- [nlm-ispc](https://github.com/AmusementClub/vs-nlm-ispc) / [nlm-cuda](https://github.com/AmusementClub/vs-nlm-cuda) / [knlmeanscl](https://github.com/Khanattila/KNLMeansCL)
- [akarin](https://github.com/AkarinVS/vapoursynth-plugin)
- [resize2](https://github.com/Jaded-Encoding-Thaumaturgy/vapoursynth-resize2)
- [fmtconv](https://gitlab.com/EleonoreMizo/fmtconv/)
- [descale](https://github.com/Jaded-Encoding-Thaumaturgy/vapoursynth-descale)
- [placebo](https://github.com/sgt0/vs-placebo)
- [awarpsharp2](https://github.com/dubhater/vapoursynth-awarpsharp2) / [warpsharpsf](https://github.com/IFeelBloated/warpsharp)
- [tcanny](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-TCanny)
- [tedgemask](https://github.com/dubhater/vapoursynth-tedgemask)
- [retinex](https://github.com/HomeOfVapourSynthEvolution/VapourSynth-Retinex)
- [adaptivegrain](https://github.com/Irrational-Encoding-Wizardry/adaptivegrain)
- [rgvs](https://github.com/vapoursynth/vs-removegrain) / [rgsf](https://github.com/IFeelBloated/RGSF)
- [zsmooth](https://github.com/adworacz/zsmooth)
- [vszip](https://github.com/dnjulek/vapoursynth-zip)
- [bilateralgpu](https://github.com/WolframRhodium/VapourSynth-BilateralGPU)
- [dpid](https://github.com/WolframRhodium/VapourSynth-dpid)
- [bestsource](https://github.com/vapoursynth/bestsource)
- [dgdecnv](https://www.rationalqm.us/dgdecnv/dgdecnv.html)
- [d2vsource](https://github.com/dwbuiten/d2vsource)
- [dvdsrc2](https://github.com/jsaowji/dvdsrc2)
- [ffms2](https://github.com/FFMS/ffms2)
- [lsmas](https://github.com/HomeOfAviSynthPlusEvolution/L-SMASH-Works)