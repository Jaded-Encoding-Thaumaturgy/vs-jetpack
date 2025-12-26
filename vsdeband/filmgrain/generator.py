import math
from dataclasses import dataclass
from typing import Self

from vsexprtools import inline_expr
from vsexprtools.inline.helpers import ClipVar, ComputedVar
from vsexprtools.inline.manager import InlineExprWrapper
from vsrgtools import remove_grain
from vstools import (
    core,
    get_neutral_values,
    get_u,
    get_v,
    get_y,
    join,
    vs,
)

from ..noise import GrainFactoryBicubic
from .config import (
    FilmGrainConfig,
    GrainConfig,
    MergeType,
)

__all__ = [
    "FilmGrainProcessor",
    "GrainLayerFactory"
]


@dataclass
class Chroma:
    u: float
    v: float

    def clamp(self, min_value: float, max_value: float) -> Self:
        self.u = max(min_value, min(max_value, self.u))
        self.v = max(min_value, min(max_value, self.v))
        return self


class GrainLayerFactory:
    @staticmethod
    def create(
        clip: vs.VideoNode,
        config: GrainConfig,
        seed: int,
        deterministic: bool = True,
        identifier: int | None = None,
    ) -> vs.VideoNode:

        blank = clip.std.BlankClip(keep=True, color=get_neutral_values(clip))

        grain_clip = config.grainer(
            blank,
            strength=config.strength,
            static=False,
            scale=config.size,
            temporal=config.temporal_blur,
            scaler=GrainFactoryBicubic(config.sharpness),
            seed=(seed if deterministic else -1),
            neutral_out=True,
            protect_neutral_chroma=False,
        )

        return grain_clip


class PostFilter:
    def __init__(self, config: FilmGrainConfig):
        self.config = config

    def emulsion(self, clip: vs.VideoNode) -> vs.VideoNode:
        return clip

    def path_to_white(self, clip: vs.VideoNode) -> vs.VideoNode:
        return clip


class Prefilter:
    def __init__(self, config: FilmGrainConfig):
        self.config = config

    def spatial(self, clip: vs.VideoNode) -> vs.VideoNode:
        if self.config.preblur is False:
            return clip

        try:
            w = clip.width
            base_strength = self.config.merge.max_strength_y
            expr = (base_strength ** 0.318 - 0.61) * (max(w / 1920.0 - 0.46, 0.0) ** 0.4 + 0.2) * self.config.preblur

            if expr > 0.35:
                return core.tcanny.TCanny(clip, expr, mode=-1)

            blur_amount = math.sqrt(self.config.preblur) / 2.0 * math.sqrt(2.0)

            if blur_amount > 0.35:
                return core.tcanny.TCanny(clip, blur_amount, mode=-1)

        except Exception:
            pass

        return clip

    def temporal(self, grain_layer: vs.VideoNode) -> vs.VideoNode:

        if not self.config.temporal_blur:
            return grain_layer

        if isinstance(self.config.temporal_blur, tuple):
            tb_amount, _ = self.config.temporal_blur
        else:
            tb_amount = float(self.config.temporal_blur)

        temporal = grain_layer.zsmooth.TemporalSoften(1, 255, 0, 255)
        return core.std.Merge(grain_layer, temporal, weight=tb_amount)


class FilmGrainProcessor(Prefilter, PostFilter):
    def __init__(self, config: FilmGrainConfig):
        super().__init__(config)
        self.config = config
        self.factory = GrainLayerFactory()
        self.post_filter = PostFilter(config)
        self.prefilter = Prefilter(config)

    def process(self, clip: vs.VideoNode, debug: bool = False) -> vs.VideoNode:

        if self.config.fast:
            return self._process_fast(clip, debug)

        th1, th2, th3, th4 = self.config.thresholds.to_scaled(clip)

        scaled = self.config.resolution_policy.apply(
            clip,
            self.config.merge,
            self.config.size,
            self.config.sharpness,
            self.config.bump,
            self.config.strength,
            self.config.s16mm,
        )

        strengths, size, sharpness, bump = scaled.strengths, scaled.size, scaled.sharpness, scaled.bump
        gen_y, gen_c = scaled.gen_strength

        base_gcfg = GrainConfig(
            strength=(gen_y * 10, gen_c * 10),
            sharpness=sharpness,
            size=size,
            grainer=self.config.grainer,
            blur=True,
            bump=bump,
        )

        grain_layer = self.factory.create(clip, base_gcfg, self.config.seed, self.config.deterministic)
        grain_layer = remove_grain(grain_layer, 12)
        grain_layer = self.prefilter.temporal(grain_layer)

        preblur_clip = self.prefilter.spatial(clip)

        with inline_expr([preblur_clip, grain_layer]) as ie:
            x, grain = ie.vars

            normalized = x / x.LumaMax

            match self.config.merge_type:
                case MergeType.GAMMA:
                    xs = x
                case MergeType.LINEAR:
                    xs = (normalized ** 2.2) * x.RangeMax
                case MergeType.LOG:
                    xs = (normalized ** 0.454545) * x.RangeMax

            if self.config.bump != 0.0:
                neighbor = ie.op.rel_pix(grain, -1, 1)
                grain = grain + (neighbor - grain) * self.config.bump

            w_dark, w_mid, w_bright = self.generate_ramp(xs, th1, th2, th3, th4)

            applied = (grain - x.Neutral)

            mult_ramp_y = w_dark * strengths.dark.y + w_mid * strengths.mid.y + w_bright * strengths.bright.y

            if any(weight != 1.0 for weight in (strengths.dark.u, strengths.mid.u, strengths.bright.u)):
                mult_ramp_u = w_dark * strengths.dark.u + w_mid * strengths.mid.u + w_bright * strengths.bright.u
                mult_ramp_v = w_dark * strengths.dark.v + w_mid * strengths.mid.v + w_bright * strengths.bright.v

                ie.out.y = x + applied * mult_ramp_y
                ie.out.u = x + applied * mult_ramp_u
                ie.out.v = x + applied * mult_ramp_v

            ie.out = x + applied * mult_ramp_y

        result = ie.clip

        if debug:
            return self._debug(result)

        return result

    def _process_fast(self, clip: vs.VideoNode, debug: bool) -> vs.VideoNode:

        th1, th2, th3, th4 = self.config.thresholds.to_scaled(clip)

        scaled = self.config.resolution_policy.apply(
            clip,
            self.config.merge,
            self.config.size,
            self.config.sharpness,
            self.config.bump,
            self.config.strength,
            self.config.s16mm,
        )

        strengths, size, sharpness, bump = scaled.strengths, scaled.size, scaled.sharpness, scaled.bump
        gen_y, gen_c = scaled.gen_strength

        y_in = get_y(clip)
        y_pre = self.prefilter.spatial(y_in)

        base_gcfg = GrainConfig(
            strength=gen_y,
            sharpness=sharpness,
            size=size,
            grainer=self.config.grainer,
            blur=True,
            bump=bump,
        )

        grain_y = self.factory.create(y_in, base_gcfg, self.config.seed, self.config.deterministic)
        grain_y = remove_grain(grain_y, 12)
        grain_y = self.prefilter.temporal(grain_y)

        grain_yuv = self.y_to_yyy(clip, grain_y)
        pre_in = join(y_pre, get_u(clip), get_v(clip), vs.ColorFamily.YUV)

        chroma = Chroma(u=strengths.mid.u, v=strengths.mid.v)

        with inline_expr([pre_in, grain_yuv]) as ie:
            x, grain = ie.vars

            neutral = x.Neutral
            normalized = x / x.LumaMax

            match self.config.merge_type:
                case MergeType.GAMMA:
                    xs = x
                case MergeType.LINEAR:
                    xs = (normalized ** 2.2) * x.RangeMax
                case MergeType.LOG:
                    xs = (normalized ** 0.454545) * x.RangeMax

            if self.config.bump != 0.0:
                neighbor = ie.op.rel_pix(grain, -1, 1)
                grain = grain + (neighbor - grain) * self.config.bump

            w_dark, w_mid, w_bright = self.generate_ramp(xs, th1, th2, th3, th4)

            mult_ramp_y = w_dark * strengths.dark.y + w_mid * strengths.mid.y + w_bright * strengths.bright.y

            applied = (grain - neutral)

            # Adjust chroma amplitude by generator C/Y ratio
            uv_factor = 1.0 if gen_y == 0 else (gen_c / gen_y)

            if chroma.v == 1.0 and chroma.u == 1.0 and uv_factor == 1.0:
                ie.out = x + applied * mult_ramp_y
            else:
                ie.out.y = x + applied * mult_ramp_y
                u_applied = applied * uv_factor
                v_applied = applied * uv_factor
                ie.out.u = (x + u_applied) if chroma.u == 1.0 else (x + u_applied * chroma.u)
                ie.out.v = (x + v_applied) if chroma.v == 1.0 else (x + v_applied * chroma.v)

        result = ie.clip

        if debug:
            return self._debug(result)

        return result

    def y_to_yyy(self, clip: vs.VideoNode, grain_y: vs.VideoNode) -> vs.VideoNode:

        fmt = clip.format
        assert fmt is not None

        ssx, ssy = fmt.subsampling_w, fmt.subsampling_h

        u_w = clip.width >> ssx
        u_h = clip.height >> ssy

        grain_u = grain_y.resize.Point(width=u_w, height=u_h)
        grain_v = grain_u

        return join(grain_y, grain_u, grain_v, vs.ColorFamily.YUV)


    def generate_ramp(
            self, xs: ClipVar | ComputedVar,
            th1: float, th2: float, th3: float, th4: float
        ) -> tuple[ComputedVar, ComputedVar, ComputedVar]:

        ie = InlineExprWrapper

        t1 = (xs - th1) / (th2 - th1)
        t2 = (xs - th3) / (th4 - th3)

        w_dark = ie.op.tern(xs < th1, 1.0, ie.op.tern(xs < th2, 1.0 - t1, 0.0))
        w_mid = ie.op.tern(xs < th1, 0.0, ie.op.tern(
            xs < th2, t1, ie.op.tern(xs < th3, 1.0, ie.op.tern(xs < th4, 1.0 - t2, 0.0))))
        w_bright = ie.op.tern(xs < th3, 0.0, ie.op.tern(xs < th4, t2, 1.0))

        return w_dark, w_mid, w_bright

    def _debug(
            self, clip: vs.VideoNode
        ) -> vs.VideoNode:

        debug_text = "\n".join([
            f"strength: {self.config.strength}",
            f"grainer: {self.config.grainer}",
            f"temporal_blur: {self.config.temporal_blur}",
            f"preblur: {self.config.preblur}",
            f"bump: {self.config.bump}",
            f"deterministic: {self.config.deterministic}",
            f"thresholds: {self.config.thresholds}",
            f"resolution_policy: {self.config.resolution_policy}",
            f"seed: {self.config.seed}",
            f"grain_size: {self.config.size}",
            f"grain_sharpness: {self.config.sharpness}",
            f"dark: {self.config.merge.dark}",
            f"mid: {self.config.merge.mid}",
            f"bright: {self.config.merge.bright}",
            f"merge_type: {self.config.merge_type}",
            f"fast: {self.config.fast}",
        ])

        return clip.text.Text(text=debug_text)
