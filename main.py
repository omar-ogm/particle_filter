import cv2
import backgroundSubtraction
import particle_filter
import particle
import argparse
import glob
import os

def preprocessing1(_image):
    """
    Preprocessing method, to retrieve the mask in the first dataset. The one with the ball falling called
    "SecuenciaPelota"
    :return:
    """
    BACKGROUND_IMAGE_PATH = r"resources/SecuenciaPelota/01.jpg"

    # Obtain the mask
    bck_sub = backgroundSubtraction.BackgroundSubtraction(background=cv2.imread(BACKGROUND_IMAGE_PATH), threshold=50)
    _mask = bck_sub.static_subtraction(_image)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    _mask = cv2.morphologyEx(_mask, cv2.MORPH_OPEN, kernel_small)
    return _mask

if __name__ == '__main__':

    #******* Argument retrieval *******
    # Reading arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", dest="input_path", type=str, required=True,
                        help="Path to the directory where the images of the sequence is")

    parser.add_argument("-p", "--particles", dest="particles", type=int, required=True,
                        help="Number of particles desired")

    parser.add_argument("-t", "--type", dest="type", type=str, required=True, choices=["LOCATION", "SPEED"],
                        help="LOCATION for simple particles fixed size that move randomly or \n"
                             "SPEED for particles that modify their size and also try to predict movement")

    args = vars(parser.parse_args())
    input_path = args['input_path']

    # ******* VARIABLES *******
    PARTICLE_NUMBER = args['particles']

    type = args['type']
    if (type == "LOCATION"):
        PARTICLE_TYPE = particle.ParticleType.LOCATION
    elif (type == "SPEED"):
        PARTICLE_TYPE = particle.ParticleType.LOCATION_BBOX_SPEED
    NEIGHBOURHOOD_SIZE = (10, 10)

    # Iterate over all the images of the directory.
    all_files_sorted = sorted(glob.glob(os.path.join(input_path, '*')))

    first_execution = True
    for image_path in all_files_sorted:

        image = cv2.imread(image_path)
        image_size = image.shape

        mask = preprocessing1(image)

        #Debug
        cv2.imshow("Mask", mask)
        # cv2.waitKey(0)

        if (first_execution):
            first_execution = False
            # A particle filter instance
            pF = particle_filter.ParticleFilter(particle_number=PARTICLE_NUMBER, image_size=image_size,
                                                particle_type=PARTICLE_TYPE, neighbourhood_size=NEIGHBOURHOOD_SIZE)

        pF.execute(image, mask=mask, debug_mode=True)


    # TEST
    # Simulate iteration with the same image

    # pF.execute(image, mask=mask, debug_mode=True)
    # pF.execute(image, mask=mask, debug_mode=True)
    # pF.execute(image, mask=mask, debug_mode=True)
