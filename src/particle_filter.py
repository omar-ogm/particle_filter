import numpy as np
import cv2
from particle import ParticleType, ParticleLocation


class ParticleFilter:
    def __init__(self, particle_number, image_size, particle_type, neighbourhood_size):
        """
        :param particle_number: The number of particles that the algorithm will handle
        :param image_size: A tuple with (height, width, channels) as tupla
        :param particle_type: Type of particle.
        :param neighbourhood_size: A tupla of two int elements that represents the size of the rectangle around the
        particles, that will be used to calculate the weight in that particle.
        """
        self.image_size = image_size
        self.particle_number = particle_number
        self.particles_list = []
        self.particle_type = particle_type
        self.neighbourhood_size = neighbourhood_size
        self.actual_image = None

        self.is_first_execution = True
        self.total_weight = 0

    def execute(self, image, mask, debug_mode=False):
        """
        Full execution of the iterative method of the particle filter, until all the steps are completed and the
        algorithm is considered to have converge
        :param image: the new image to apply the particle filter on
        :param mask: A binary mask of the image, where True pixels are the objects we want the particles to focus on
        :param debug_mode: True will show the particles on the image, False otherwise
        :return:
        """
        self.actual_image = image

        while (self.__is_initialization_needed()):
            self.__initialization()
            self.__evaluation(mask=mask) # Evaluation updates total_weight value use in is_initialization_needed

        if (debug_mode):
            self.__draw_particles(draw_points=True, draw_bounding_box=True)

        # Get the estimation for all the parameters of the state. x,y, vx, vy, Ix,Iy and all the ones you can think of.
        estimation_state = self.__estimation(use_mean_value=False)

        # Show the estimation
        self.__draw_estimation(estimation_state)

    def __initialization(self):
        """
        The initialization of the particles over the image. Just for the first frame or in case the object being
        tracked is lost, goes out of scope of the particles area.
        The initialization used is based on a uniform pdf over the image pixel.
        :return:
        """
        height, width, _ = self.image_size #CHEQUEAR QUE DE VERDAD SEA ASI Y NO AL REVES
        x_positions = np.random.randint(0, width + 1, self.particle_number)
        y_positions = np.random.randint(0, height + 1, self.particle_number)
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

        print("Initializing particles")

    def __is_initialization_needed(self):
        """
        Checks if the particles has been already initialized, or if the first initialization wasnt good enough to
        restart it.
        True if the object was capture by some particles, surpassing the threshold defined for the total weight
        False if the total_weight obtained is lower than the threshold define, meaning a reinitialization is needed.
        :return: (Bool) True upon initialization needed, False otherwise
        """
        result = False
        total_weight_threshold = 200

        if (self.is_first_execution):
            self.is_first_execution = False
            result = True
        elif (self.total_weight < total_weight_threshold):
            result = True
            print("Reinitialization needed due to low weight")

        return result



    def __evaluation(self, mask):
        """
        The evaluation is the steep where the weights for each particle is computed. The evaluation receives a boolean
        mask. Each True pixel of the boolean mask that is in the neighbourhood of a particle adds weigh to that particle
        :param mask: A binary mask of the image, where True pixels are the objects we want the particles to focus on
        :return:
        """
        self.total_weight = 0
        for particle in self.particles_list:
            top_left_point = np.array([particle.x - self.neighbourhood_size[0], particle.y - self.neighbourhood_size[1]])
            bottom_right_point = np.array([particle.x + self.neighbourhood_size[0], particle.y + self.neighbourhood_size[1]])
            # If there are points with negative values, change them to 0
            top_left_point = top_left_point.clip(min=0)
            bottom_right_point = bottom_right_point.clip(min=0)
            # NOTE: In case the area goes outside the limits of the image, the "roi" will be smaller than the value it
            # should, not a problem though
            roi = mask[top_left_point[1]:bottom_right_point[1], top_left_point[0]:bottom_right_point[0]]
            particle.weight = np.count_nonzero(roi)
            self.total_weight += particle.weight

            # print(roi.size) #Total number of elements, in this case pixels
            # print(top_left_point)
            # print(bottom_right_point)
            # cv2.imshow("test", roi)
            # cv2.waitKey(0)

        # Normalize weights values for all particles, based on the total pixels count of all the particles
        # test_weight = 0 # JUST FOR TESTING THAT IT EQUALS 1 AFTER ALL

        if (self.total_weight != 0):

            for particle in self.particles_list:
                particle.weight = particle.weight / self.total_weight
                # test_weight += particle.weight

            # print(test_weight)

    def __estimation(self, use_mean_value=False, mean_number=0):
        """
        Estimate the values of the state. Different criteria can be applied here.
        :param use_mean_value: True if the mean value of the states among n_number of the particles with higher weights
        will be used. False the state of the particle with the highest weight will be used.
        :param mean_number: Leave as 0, all the particles will be used to do the mean. If different from 0 this will be
        the number of particles used to measure the mean. The particles selected will be those of higher weight value.
        :return: The estimation, a list with all the parameters of the state. [x, y]
        """
        estimation = []

        if (use_mean_value):
            #TODO MEAN
            print("TODO")
        else:
            weights_list = [particle.weight for particle in self.particles_list]
            most_possible_particle = self.particles_list[np.argmax(weights_list)]
            estimation = [most_possible_particle.x, most_possible_particle.y]

        return estimation


    def __draw_estimation(self, central_coord):
        image = np.copy(self.actual_image)
        cv2.circle(image, center=tuple(central_coord), radius=1, color=(0, 255, 255), thickness=1)
        top_left_point = (central_coord[0] - self.neighbourhood_size[0], central_coord[1] - self.neighbourhood_size[1])
        bottom_right_point = (central_coord[0] + self.neighbourhood_size[0], central_coord[1] + self.neighbourhood_size[1])
        cv2.rectangle(image, pt1=top_left_point, pt2=bottom_right_point, color=(255, 255, 0),
                      thickness=1)
        cv2.imshow("Estimation", image)
        cv2.waitKey(0)

    def __draw_particles(self, draw_points, draw_bounding_box):
        image = np.copy(self.actual_image)
        for particle in self.particles_list:
            position = (particle.x, particle.y)
            if (draw_points):
                cv2.circle(image, center=position, radius=1, color=(0, 255, 0), thickness=1)
            if (draw_bounding_box):
                top_left_point = (particle.x - self.neighbourhood_size[0], particle.y - self.neighbourhood_size[1])
                bottom_right_point = (particle.x + self.neighbourhood_size[0], particle.y + self.neighbourhood_size[1])
                cv2.rectangle(image, pt1=top_left_point, pt2=bottom_right_point, color=(255, 0, 0),
                              thickness=1)

        cv2.imshow("Particle Filter", image)
        cv2.waitKey(0)
