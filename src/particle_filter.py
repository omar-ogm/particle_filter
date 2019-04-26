import numpy as np
import cv2
from particle import ParticleType, ParticleLocation, ParticleLocationSizeSpeed
import copy


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
        self.max_initialization_iters = 5

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
        evaluation_done = False

        initialization_iter = 0
        # 1-2. INITIALIZATION AND EVALUATION
        while (self.__is_initialization_needed()):
            self.__initialization()
            self.__evaluation(mask=mask)  # Evaluation updates total_weight value use in is_initialization_needed
            evaluation_done = True
            initialization_iter += 1

            if (initialization_iter == self.max_initialization_iters):
                print("The initialization could not find the object, object is missing!")
                return

        # 2. EVALUATION
        if (not evaluation_done):
            self.__evaluation(mask=mask)


        if (debug_mode):
            self.__draw_particles(draw_points=True, draw_bounding_box=True)

        # 3. ESTIMATION
        # Get the estimation for all the parameters of the state. x,y, vx, vy, Ix,Iy and all the ones you can think of.
        estimation_state = self.__estimation(use_mean_value=False)

        # Show the estimation
        self.__draw_estimation(estimation_state)

        # 4. RESAMPLING or SELECTION
        self.__resampling()

        # 5. DIFFUSION
        self.__diffusion()

        # 6. MOVEMENT PREDICTION
        if self.particle_type == ParticleType.LOCATION_SPEED or self.particle_type == ParticleType.LOCATION_BBOX_SPEED:
            self.__movement_prediction()

        # Show for a period of time all each image
        cv2.waitKey(500)  # 0.5 secs

    def __initialization(self):
        """
        The initialization of the particles over the image. Just for the first frame or in case the object being
        tracked is lost, goes out of scope of the particles area.
        The initialization used is based on a uniform pdf over the image pixel.
        :return:
        """
        self.particles_list = []
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
            elif self.particle_type == ParticleType.LOCATION_BBOX:
                print("Not implemented ParticleType.LOCATION_BBOX, exiting")
                exit(-1)
            elif self.particle_type == ParticleType.LOCATION_SPEED:
                print("Not implemented ParticleType.LOCATION_SPEED, exiting")
                exit(-1)
            elif self.particle_type == ParticleType.LOCATION_BBOX_SPEED:
                Ix = self.neighbourhood_size[0]
                Iy = self.neighbourhood_size[1]
                Vx = 0
                Vy = 0
                state = (position[0], position[1], Ix, Iy, Vx, Vy)
                particle = ParticleLocationSizeSpeed(state=state, weight=weight)
            else:
                print("Not a valid particle filter type, exiting")
                exit(-1)

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
            if self.particle_type == ParticleType.LOCATION or self.particle_type == ParticleType.LOCATION_SPEED:
                top_left_point = np.array(
                    [particle.x - self.neighbourhood_size[0], particle.y - self.neighbourhood_size[1]])
                bottom_right_point = np.array(
                    [particle.x + self.neighbourhood_size[0], particle.y + self.neighbourhood_size[1]])
                # If there are points with negative values, change them to 0
                top_left_point = top_left_point.clip(min=0)
                bottom_right_point = bottom_right_point.clip(min=0)
                # NOTE: In case the area goes outside the limits of the image, the "roi" will be smaller than the value
                # it should, not a problem though
                roi = mask[top_left_point[1]:bottom_right_point[1], top_left_point[0]:bottom_right_point[0]]
                particle.weight = np.count_nonzero(roi)
                self.total_weight += particle.weight
            elif self.particle_type == ParticleType.LOCATION_BBOX or \
                    self.particle_type == ParticleType.LOCATION_BBOX_SPEED:
                top_left_point = np.array([particle.x - particle.Ix, particle.y - particle.Iy])
                bottom_right_point = np.array([particle.x + particle.Ix, particle.y + particle.Iy])
                # If there are points with negative values, change them to 0
                top_left_point = top_left_point.clip(min=0)
                bottom_right_point = bottom_right_point.clip(min=0)
                # NOTE: In case the area goes outside the limits of the image, the "roi" will be smaller than the value
                # it should, not a problem though
                roi = mask[top_left_point[1]:bottom_right_point[1], top_left_point[0]:bottom_right_point[0]]

                # The particle weight will be defined by the area with pixels from the target on it and also by the
                # pixels that dont belong to the target, otherwise the Ix,Iy will grow indefinitely so it always get the
                # full target, so adding a penalization based on the pixel not in the target I made sure it tries to
                # cover only the target.
                # TODO: Could be improved using something like IoU with using the groundtruth the number of pixels
                #  from the target.
                pixels_on_target = np.count_nonzero(roi)
                total_pixels = roi.size
                particle.weight = pixels_on_target - (total_pixels-pixels_on_target)*0.75
                if (particle.weight < 0):
                    particle.weight = 0
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
            estimation = most_possible_particle

        return estimation

    def __resampling(self):
        """
        The selection or resampling is the technique used to resample the particles based or their weights. Like in
        evolution the particles with more possibilities to survive are the ones with the higher weights. To select the
        new population of particles a sampling with replacement will be used. There will be sample as many particles as
        the original population had
        :return:
        """
        # sort the list in place in ascending order by default. use reverse to inverse the order.
        self.particles_list.sort(key=lambda particle: particle.weight) # From lower to upper
        # Get the list of weights only, ordered.
        weights_list = np.asarray([particle.weight for particle in self.particles_list])
        # Now the the weights are ordered, the accumulated weight list must be created
        weights_accumulated_list = np.cumsum(weights_list)
        new_particles_list = []

        # Roullete method or Roullete selection in genetic algorithms (works the same, the best will survive while the
        # worst will perish in the end)
        for idx in range(0, len(self.particles_list)):
            random = np.random.uniform()  # value between 0-1 (1 is excluded)
            # max in case of several maximum(True in this case) will return the first weight encountered that is bigger.
            particle_idx = np.argmax(weights_accumulated_list > random)
            # Its important here to make a deepcopy so the object in the new list is not a reference but a full copy on
            # the object. So changes in teh particles list wont affect the new_list.
            new_particles_list.append(copy.deepcopy(self.particles_list[particle_idx]))

        self.particles_list = new_particles_list  # Update the list of particles with the new values.

    def __diffusion(self):
        """
        The diffusion step, is apply after the new population of particles has been chosen. Since the particles chosen
        are repeated, a perturbation or diffusion is applied to evade the impoverishment of the sample. Like in
        evolution the new population should be made of the genes of the stronger individuals in the previous population
        but also had their unique distinctiveness.
        :return:
        """
        sigma = 10  #standard deviation in pixels. Is a experimental value.
        sigma_size = 1
        sigma_speed = 5

        for particle in self.particles_list:
            # TODO: Maybe is better to have the particles made its own diffusion in a class method
            # Common for all particles
            particle.x = int(np.round(np.random.normal(particle.x, sigma)))
            particle.y = int(np.round(np.random.normal(particle.y, sigma)))

            if self.particle_type == ParticleType.LOCATION:
                continue
            elif self.particle_type == ParticleType.LOCATION_SPEED:
                raise NotImplementedError()
            elif self.particle_type == ParticleType.LOCATION_BBOX:
                raise NotImplementedError()
            elif self.particle_type == ParticleType.LOCATION_BBOX_SPEED:
                particle.Ix = int(np.round(np.random.normal(particle.Ix, sigma_size)))
                particle.Iy = int(np.round(np.random.normal(particle.Iy, sigma_size)))
                particle.Vx = int(np.round(np.random.normal(particle.Vx, sigma_speed)))
                particle.Vy = int(np.round(np.random.normal(particle.Vy, sigma_speed)))

    def __movement_prediction(self):
        """
        This method simulates the movement of the particles. The particles had a speed, so we move the particles by
        their speed to simulate the movement allowing the best particles to follow the object better
        :return:
        """
        for particle in self.particles_list:
            if self.particle_type == ParticleType.LOCATION or self.particle_type == ParticleType.LOCATION_BBOX:
                return
            elif self.particle_type == ParticleType.LOCATION_SPEED or self.particle_type == ParticleType.LOCATION_BBOX_SPEED:
                particle.x = particle.x + round(particle.Vx)
                particle.y = particle.y + round(particle.Vy)

    def __draw_estimation(self, particle):
        image = np.copy(self.actual_image)
        cv2.circle(image, center=(particle.x, particle.y), radius=1, color=(0, 255, 255), thickness=1)

        if self.particle_type == ParticleType.LOCATION or self.particle_type == ParticleType.LOCATION_SPEED:
            top_left_point = (particle.x - self.neighbourhood_size[0], particle.y - self.neighbourhood_size[1])
            bottom_right_point = (particle.x + self.neighbourhood_size[0], particle.y + self.neighbourhood_size[1])
        elif self.particle_type == ParticleType.LOCATION_BBOX \
                or self.particle_type == ParticleType.LOCATION_BBOX_SPEED:
            top_left_point = (particle.x - particle.Ix, particle.y - particle.Iy)
            bottom_right_point = (particle.x + particle.Ix, particle.y + particle.Iy)

        cv2.rectangle(image, pt1=top_left_point, pt2=bottom_right_point, color=(255, 255, 0),
                      thickness=1)
        if (self.particle_type == ParticleType.LOCATION_SPEED or self.particle_type == ParticleType.LOCATION_BBOX_SPEED):
            cv2.putText(image, "Vx=" + str(particle.Vx), (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(image, "Vy=" + str(particle.Vy), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Estimation", image)

    def __draw_particles(self, draw_points, draw_bounding_box):
        image = np.copy(self.actual_image)
        for particle in self.particles_list:
            position = (particle.x, particle.y)
            if (draw_points):
                cv2.circle(image, center=position, radius=1, color=(0, 255, 0), thickness=1)
            if (draw_bounding_box):
                if self.particle_type == ParticleType.LOCATION or self.particle_type == ParticleType.LOCATION_SPEED:
                    top_left_point = (particle.x - self.neighbourhood_size[0], particle.y - self.neighbourhood_size[1])
                    bottom_right_point = (particle.x + self.neighbourhood_size[0], particle.y + self.neighbourhood_size[1])
                elif self.particle_type == ParticleType.LOCATION_BBOX \
                        or self.particle_type == ParticleType.LOCATION_BBOX_SPEED:
                    top_left_point = (particle.x - particle.Ix, particle.y - particle.Iy)
                    bottom_right_point = (particle.x + particle.Ix, particle.y + particle.Iy)

                cv2.rectangle(image, pt1=top_left_point, pt2=bottom_right_point, color=(255, 0, 0), thickness=1)

        cv2.imshow("Particles", image)
