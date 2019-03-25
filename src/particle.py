from abc import ABC, abstractmethod
from enum import Enum

class Particle(ABC):
    """
    Base class to define a particle for a particle filter.
    Has an state and a weight. The state can be form by several parameters.
    """
    @abstractmethod
    def __init__(self, state, weight):
        raise NotImplementedError

class ParticleLocation(Particle):
    """
    A particle that has as parameters x and y to locate objects.
    """
    def __init__(self, state, weight):
        self.x, self.y = state
        self.weight = weight

class ParticleLocationSize(Particle):
    """
    A particle that has as parameters x and y to locate the centroid of objects and a boundind box.
    """
    def __init__(self, state, weight):
        self.x = state[0]
        self.y = state[1]
        self.Ix = state[2]
        self.Iy = state[3]
        self.weight = weight

class ParticleLocationSpeed(Particle):
    """
    A particle that has as parameters x and y to locate objects and estimate the speed.
    """
    def __init__(self, state, weight):
        self.x = state[0]
        self.y = state[1]
        self.Vx = state[2]
        self.Vy = state[3]
        self.weight = weight

class ParticleLocationSizeSpeed(Particle):
    """
    A particle that has as parameters x and y to locate objects, the bounding box and estimate the speed
    """
    def __init__(self, state, weight):
        self.x = state[0]
        self.y = state[1]
        self.Ix = state[2]
        self.Iy = state[3]
        self.Vx = state[4]
        self.Vy = state[5]
        self.weight = weight

class ParticleType(Enum):
    LOCATION = 0
    LOCATION_SPEED = 1
    LOCATION_BBOX = 2
    LOCATION_BBOX_SPEED = 3
