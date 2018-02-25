import numpy as np
from IPython import embed
from keras.preprocessing.image import load_img, img_to_array


def get_image_array(fname, config):
    image = load_img(fname, target_size=(config["img_x"], config["img_y"]))
    image = img_to_array(image)
    image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
    return image


def get_perturbed_images(image, perturbations):
    perturbed_images = list()
    for candidate in perturbations:
        perturbed_image = np.copy(image)
        for p in candidate:
            x = int(p[0])
            y = int(p[1])
            perturbed_image[0][x][y][0] = p[2]
            perturbed_image[0][x][y][1] = p[3]
            perturbed_image[0][x][y][2] = p[4]
        perturbed_images.append(perturbed_image)
    perturbed_images_arr = np.concatenate(perturbed_images)
    return perturbed_images_arr


def get_probability_for_class(predictions, class_name):
    """

    Args:
        predictions (list[tuple]):
        class_name (str):

    Returns:
        float
    """
    for cls in predictions:
        if str(cls[1]) == class_name:
            return float(cls[2])
    print "Class not found in predictions"
    embed()
