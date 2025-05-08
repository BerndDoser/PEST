from dataclasses import dataclass, field
from enum import Enum


class OrientationType(Enum):
    ORIGINAL = 1
    FACE_ON = 2
    EDGE_ON = 3
    RANDOM = 4


class PropertyType(Enum):
    STELLAR_MASS = 1
    STAR_FORMATION_RATE = 2
    METALLICITY = 3
    GAS_DENSITY = 4
    DARK_MATTER_DENSITY = 5
    DARK_MATTER_VELOCITY = 6
    GAS_VELOCITY = 7
    STELLAR_VELOCITY = 8
    STAR_AGE = 9
    STAR_METALLICITY = 10


@dataclass
class Selector:
    property: PropertyType
    min_value: float
    max_value: float


@dataclass
class IllustrisGenerator:
    sim: str = "TNG50-1"
    selectors: list[Selector] = field(default_factory=list)
    # component = ("stars",)
    # objects = ("centrals",)
    # field = ("Masses",)
    # fov = ("scaled",)  # [kpc]
    # image_depth = (1.0,)  #  1 particles per pixel (min. S/N=sqrt(depth))
    # image_size = (128,)
    # smoothing = (0.0,)  # [kpc]
    # image_scale = ("log",)
    orientation: OrientationType = OrientationType.RANDOM
    # output_path = ("./images_test_local/",)


ig = IllustrisGenerator(selectors=[Selector(PropertyType.STELLAR_MASS, 5e10, 5.2e10)])
print(ig)
