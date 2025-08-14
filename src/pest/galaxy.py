from dataclasses import dataclass, field
from typing import List


@dataclass
class Particle:
    """Particle properties"""

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
    """Star-specific properties"""

    luminosity: float = 1.0


@dataclass
class Gas(Particle):
    """Gas-specific properties"""

    temperature: float = 100.0


@dataclass
class Galaxy:
    id: int
    central: bool
    mass: float
    position: List[float]
    velocity: List[float]
    stars: List[Star] = field(default_factory=list)
    gas: List[Gas] = field(default_factory=list)

    def __post_init__(self):
        if len(self.position) != 3:
            raise ValueError("Position must have exactly 3 coordinates")
        if len(self.velocity) != 3:
            raise ValueError("Velocity must have exactly 3 coordinates")
