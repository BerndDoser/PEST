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
class Star(Particle):
    luminosity: float = 1.0  # Example of a star-specific property


@dataclass
class Gas(Particle):
    temperature: float = 100.0  # Example of a gas-specific property


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


star = Star(id=1, mass=1.0, position=[0.0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0], luminosity=10.0)
gas = Gas(id=2, mass=0.5, position=[1.0, 0.0, 0.0], velocity=[0.0, 0.1, 0.0], temperature=500.0)

galaxy_1 = Galaxy(
    id=1,
    central=True,
    mass=1.0,
    position=[0.0, 0.0, 0.0],
    velocity=[0.0, 0.0, 0.0],
    particles=[star, gas],
)
galaxy_2 = Galaxy(
    id=2,
    central=True,
    mass=2.0,
    position=[0.0, 0.0, 1.0],
    velocity=[0.0, 0.0, 0.0],
    particles=[],
)

# type alias
Galaxies = List[Galaxy]

galaxies: Galaxies = [galaxy_1, galaxy_2]

print(galaxies)
