import particle_filter


if __name__ == '__main__':
    PARTICLE_NUMBER = 100
    PARTICLE_TYPE = particle_filter.ParticleType.LOCATION


    #TODO: read images from location.
    particle_filter.ParticleFilter(particle_number=PARTICLE_NUMBER, image_size=, particle_type=PARTICLE_TYPE)