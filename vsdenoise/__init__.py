"""
Wrappers for denoising and motion-compensation-related plugins and functions.


## Example usage

```py
from vsdenoise import MVToolsPreset, Prefilter, mc_degrain, bm3d, nl_means

clip = ...

ref = mc_degrain(clip, prefilter=Prefilter.DFTTEST(), preset=MVToolsPreset.HQ_SAD, thsad=100)

denoise = bm3d(clip, sigma=0.8, tr=2, profile=bm3d.Profile.NORMAL, ref=ref, planes=0)

denoise = nl_means(denoise, 0.2, tr=2, ref=ref, planes=[1, 2])
```
"""

from .blockmatch import *
from .deblock import *
from .fft import *
from .freqs import *
from .funcs import *
from .mvtools import *
from .nlm import *
from .postprocess import *
from .prefilters import *
