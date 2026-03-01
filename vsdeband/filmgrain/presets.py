import enum

from ..noise import Grainer
from .config import FilmGrainConfig, MergeStrengths, PlaneStrength

__all__ = [
    "FilmGrainPreset",
]


class FilmGrainPreset(enum.Enum):

    EIGHT_MM = FilmGrainConfig(
        strength=(7.0, 0.6),
        size=2.0,
        sharpness=60,
    )
    SIXTEEN_MM = FilmGrainConfig(
        merge=MergeStrengths(
            dark=PlaneStrength(5.5, 5.5, 5.5),
            mid=PlaneStrength(4.5, 4.5, 4.5),
            bright=PlaneStrength(3.5, 3.5, 3.5)
        ),
        size=1.7,
        sharpness=66,
    )
    THIRTY_FIVE_MM = FilmGrainConfig(
        merge=MergeStrengths(
            dark=PlaneStrength(2.5, 2.5, 2.5),
            mid=PlaneStrength(2.0, 2.0, 2.0),
            bright=PlaneStrength(1.5, 1.5, 1.5)
        ),
        size=1.0,
        sharpness=80,
    )
    FILMGRAIN = FilmGrainConfig(
        size=1.5,
        sharpness=90,
    )
    GRAINFACTORY = FilmGrainConfig(
        merge=MergeStrengths(
            dark=PlaneStrength(1.0, 1.0, 1.0),
            mid=PlaneStrength(0.9, 0.9, 0.9),
            bright=PlaneStrength(0.8, 0.8, 0.8)
        ),
        size=1.0,
        sharpness=50,
    )
    DIGITAL = FilmGrainConfig(
        strength=(1.2, 0.5),
        merge=MergeStrengths(
            dark=PlaneStrength(1.0, 0.1, 0.1),
            mid=PlaneStrength(1.0, 0.1, 0.1),
            bright=PlaneStrength(1.0, 0.1, 0.1)
        ),
        size=1.0,
        sharpness=100,
    )
    VHS = FilmGrainConfig(
        merge=MergeStrengths(
            dark=PlaneStrength(4.0, 3.5, 3.0),
            mid=PlaneStrength(3.5, 3.0, 2.5),
            bright=PlaneStrength(3.0, 2.5, 2.0)
        ),
        size=1.8,
        sharpness=40,
        temporal_blur=(0.3, 0),
        grainer=Grainer.GAUSS(hcorr=0.5)
    )

    EXR_5245_50D = FilmGrainConfig(
        strength=(0.1, 0.5),
        size=0.8,
        sharpness=80,
    )
    EXR_5248_100T = FilmGrainConfig(
        strength=(1.0, 0.5),

        size=1.8,
        sharpness=90,
    )
    EXR_5293_200T = FilmGrainConfig(
        strength=(1.0, 0.5),

        size=1.0,
        sharpness=100,
    )
    EXR_5298_500T = FilmGrainConfig(
        strength=(1.0, 0.5),

        size=1.0,
        sharpness=100,
    )

    VISION_5274_200T = FilmGrainConfig(
        strength=(0.4, 0.5),
        size=0.4,
        sharpness=100,
    )
    VISION_5246_250D = FilmGrainConfig(
        strength=(0.9, 0.5),

        size=0.2,
        sharpness=100,
    )
    VISION_5277_320T = FilmGrainConfig(
        strength=(1.0, 0.5),

        size=1.0,
        sharpness=100,
    )
    VISION_5279_500T = FilmGrainConfig(
        strength=(1.4, 0.5),
        size=1.9,
        sharpness=100,
    )

    VISION2_5201_50D = FilmGrainConfig(
        strength=(1.0, 0.5),

        size=1.0,
        sharpness=100,
    )
    VISION2_5212_100T = FilmGrainConfig(
        strength=(1.0, 0.5),

        size=1.0,
        sharpness=100,
    )
    VISION2_5217_200T = FilmGrainConfig(
        strength=(0.2, 0.5),

        size=1.4,
        sharpness=90,
    )
    VISION2_5218_500T = FilmGrainConfig(
        strength=(0.1, 0.5),

        size=1.3,
        sharpness=100,
    )
    VISION3_5203_50D = FilmGrainConfig(
        strength=(1.0, 0.5),

        size=1.0,
        sharpness=100,
    )
    VISION3_5213_200T = FilmGrainConfig(
        strength=(1.0, 0.5),

        size=1.0,
        sharpness=100,
    )
    VISION3_5207_250D = FilmGrainConfig(
        strength=(1.0, 0.5),

        size=1.0,
        sharpness=100,
    )
    VISION3_5219_500T = FilmGrainConfig(
        strength=(0.1, 0.5),

        size=0.5,
        sharpness=100,
    )

    FX_214_480D = FilmGrainConfig(
        strength=(0.3, 0.5),
        size=0.8,
        sharpness=80,
    )

    FUJI_8510_64T = FilmGrainConfig(
        strength=(1.0, 0.5),
        size=1.0,
        sharpness=100,
    )
    FUJI_8520_64D = FilmGrainConfig(
        strength=(1.0, 0.5),

        size=1.0,
        sharpness=100,
    )
    FUJI_8530_125T = FilmGrainConfig(
        strength=(1.0, 0.5),

        size=1.0,
        sharpness=100,
    )
    FUJI_8550_250T = FilmGrainConfig(
        strength=(1.0, 0.5),

        size=1.0,
        sharpness=100,
    )
    FUJI_8560_250D = FilmGrainConfig(
        strength=(1.0, 0.5),

        size=1.0,
        sharpness=100,
    )
    FUJI_8570_500T = FilmGrainConfig(
        strength=(1.0, 0.5),

        size=1.0,
        sharpness=100,
    )

    SUPER_F_8522_64D = FilmGrainConfig(
        strength=(1.0, 0.5),

        size=1.0,
        sharpness=100,
    )
    SUPER_F_8532_125T = FilmGrainConfig(
        strength=(1.0, 0.5),

        size=1.0,
        sharpness=100,
    )
    SUPER_F_8552_250T = FilmGrainConfig(
        strength=(1.0, 0.5),

        size=1.0,
        sharpness=100,
    )
    SUPER_F_8562_250D = FilmGrainConfig(
        strength=(1.0, 0.5),

        size=1.0,
        sharpness=100,
    )
    SUPER_F_8582_400T = FilmGrainConfig(
        strength=(1.0, 0.5),

        size=1.0,
        sharpness=100,
    )
    SUPER_F_8572_500T = FilmGrainConfig(
        strength=(1.0, 0.5),

        size=1.0,
        sharpness=100,
    )

    ETERNA_V_8543_160T = FilmGrainConfig(
        strength=(1.0, 0.5),

        size=1.0,
        sharpness=100,
    )
    ETERNA_8563_250D = FilmGrainConfig(
        strength=(1.0, 0.5),

        size=1.0,
        sharpness=100,
    )
    ETERNA_8583_400T = FilmGrainConfig(
        strength=(1.0, 0.5),

        size=1.0,
        sharpness=100,
    )
    ETERNA_8573_500T = FilmGrainConfig(
        strength=(1.0, 0.5),

        size=1.0,
        sharpness=100,
    )
    REALA_8592_500D = FilmGrainConfig(
        strength=(1.0, 0.5),

        size=1.0,
        sharpness=100,
    )

    def to_config(self) -> FilmGrainConfig:
        return self.value

    def to_dict(self) -> dict:
        return {
            k: v.default
            for k, v in vars(self.value).items()
        }
