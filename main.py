import cv2
import backgroundSubtraction
import particle_filter
import particle

if __name__ == '__main__':
    PARTICLE_NUMBER = 100
    PARTICLE_TYPE = particle.ParticleType.LOCATION
    NEIGHBOURHOOD_SIZE = (10, 10)
    BACKGROUND_IMAGE_PATH = r"/home/omar/workspaces/python/Filtro_Particulas/Recursos_practica/SecuenciaPelota/1.jpg"


    # New image
    image_path = r"/home/omar/workspaces/python/Filtro_Particulas/Recursos_practica/SecuenciaPelota/10.jpg"

    image = cv2.imread(image_path)
    image_size = image.shape

    # Obtain the mask
    bck_sub = backgroundSubtraction.BackgroundSubtraction(background=cv2.imread(BACKGROUND_IMAGE_PATH), threshold=50)
    mask = bck_sub.static_subtraction(image)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)

    #TODO: read images from location.
    pF = particle_filter.ParticleFilter(particle_number=PARTICLE_NUMBER, image_size=image_size,
                                        particle_type=PARTICLE_TYPE, neighbourhood_size=NEIGHBOURHOOD_SIZE)
    pF.execute(image, mask=mask, debug_mode=True)
