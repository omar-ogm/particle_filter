from abc import ABC, abstractmethod
import numpy as np
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


class ParticleFilter:
    def __init__(self, particle_number, image_size, particle_type):
        self.image_size = image_size
        self.particle_number = particle_number
        self.particles_list = []
        self.particle_type = particle_type

    def __initialization(self):
        """
        The initialization of the particles over the image. Just for the first frame or in case the object being
        tracked is lost, goes out of scope of the particles area.
        The initialization used is based on a uniform pdf over the image pixel.
        :return:
        """
        height, width = self.image_size #CHEQUEAR QUE DE VERDAD SEA ASI Y NO AL REVES
        x_positions = np.random.randint(0, height + 1, self.particle_number)
        y_positions = np.random.randint(0, width + 1, self.particle_number)
        #Concatenate x and y in a position vector or something, and create the particles
        positions = list(zip(x_positions, y_positions))
        #WARNING: ZIP Returns an iterator not an element, so can be iterated only once
        weight = 1/self.particle_number
        for position in positions:
            particle = None
            if self.particle_type == ParticleType.LOCATION:
                state = position
                particle = ParticleLocation(state=state, weight=weight)
            # TODO: CREATE THE REST OF CASES FOR THE REST OF POSSIBLE PARTICLES

            self.particles_list.append(particle)




