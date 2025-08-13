from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Particle:
    id: int
    mass: float
    position: List[float]
    velocity: List[float]

    def __post_init__(self):
        if len(self.position) != 3:
            raise ValueError("Position must have exactly 3 coordinates")
        if len(self.velocity) != 3:
            raise ValueError("Velocity must have exactly 3 coordinates")


@dataclass
class Galaxy:
    id: int
    central: bool
    mass: float
    position: List[float]
    velocity: List[float]
    particles: Dict[str, List[Particle]] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.position) != 3:
            raise ValueError("Position must have exactly 3 coordinates")
        if len(self.velocity) != 3:
            raise ValueError("Velocity must have exactly 3 coordinates")


star = Particle(
    id=1,
    mass=1.0,
    position=[0.0, 0.0, 0.0],
    velocity=[0.0, 0.0, 0.0],
)

galaxy = Galaxy(
    id=1,
    central=True,
    mass=1.0,
    position=[0.0, 0.0, 0.0],
    velocity=[0.0, 0.0, 0.0],
    particles={"stars": [star]},
)

data = {"galaxies": [galaxy]}

print(data)
