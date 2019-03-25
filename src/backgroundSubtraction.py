import numpy as np
import cv2

class BackgroundSubtraction:
    """
    Class to apply background subtraction. Given an image or frame and using another image with the background the two
    images are subtract to obtain the foreground of the image.
    """
    def __init__(self, background, threshold):
        self.__backgroundImage = background
        self.__threshold = threshold
        self.__background_images_count = 0

    def static_subtraction(self, frame):
        """
        Given an image, the background is subtracted creating a mask to see the changes in the static image.
        :param frame: input image to apply static subtraction
        :return: A mask the same size of the original fram but with only one channel. True values represent foreground,
        false values background.
        """
        frame = frame.astype(np.int16)
        self.__backgroundImage = self.__backgroundImage.astype(np.int16)
        subtraction = np.abs(np.subtract(frame, self.__backgroundImage))
        subtraction_mask = (subtraction[:, :, 0] > self.__threshold) | (subtraction[:, :, 1] > self.__threshold) | \
                           (subtraction[:, :, 2] > self.__threshold)
        subtraction_mask = np.multiply(1, subtraction_mask).astype(float)

        return subtraction_mask