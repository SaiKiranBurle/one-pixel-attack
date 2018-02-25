import argparse
import logging

import imageio
import numpy as np
import yaml
from IPython import embed
from keras.preprocessing.image import load_img, img_to_array

from differential_evolution import init_population, gen_children
from models.base import get_model_from_name

CONFIG = None
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


def get_image_array(fname):
    image = load_img(fname, target_size=(CONFIG["img_x"], CONFIG["img_y"]))
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


def get_fit_population(fathers, children, fathers_predictions, children_predictions, true_class):
    """
    For non-targeted attacks, the fitness function is the probability of true class
    """
    final_population = list()
    for i in range(len(fathers_predictions)):
        p_father = get_probability_for_class(fathers_predictions[i], true_class)
        p_child = get_probability_for_class(children_predictions[i], true_class)
        if p_father > p_child:
            final_population.append(children[i])
        else:
            final_population.append(fathers[i])
    return np.array(final_population)


def find_adversary_image(image, model):
    original_predictions = model.predict(np.copy(image))
    true_label = original_predictions[0][0][1]
    true_label_probability = original_predictions[0][0][2]
    logging.info("True label: {}, Probability: {}".format(true_label, true_label_probability))

    population = init_population(CONFIG)
    for i in range(CONFIG["num_iterations"]):
        logging.info("Iteration: {}".format(i))
        perturbed_images = get_perturbed_images(image, population)
        perturbed_predictions = model.predict(np.copy(perturbed_images), top=1000)

        true_class_probabilities = map(lambda p: get_probability_for_class(p, true_label), perturbed_predictions)
        logging.info("Probabilites for true class: Min={}, Max={}".format(min(true_class_probabilities),
                                                                          max(true_class_probabilities)))
        if i % 10 == 0:
            imageio.imwrite('output/{}.jpg'.format(i),
                            perturbed_images[true_class_probabilities.index(min(true_class_probabilities))])

        population_children = gen_children(population, CONFIG)
        perturbed_images_children = get_perturbed_images(image, population_children)
        perturbed_predictions_children = model.predict(np.copy(perturbed_images_children), top=1000)

        population = get_fit_population(population, population_children,
                                        perturbed_predictions,
                                        perturbed_predictions_children,
                                        true_class=true_label)
    embed()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', dest='config_file', help='config file')
    parser.add_argument('--input', '-i', dest='input_image', help='input image file')
    args = parser.parse_args()
    # global CONFIG
    CONFIG = yaml.safe_load(open(args.config_file))
    model = get_model_from_name(CONFIG["model"])
    # model = None
    image_arr = get_image_array(args.input_image)
    find_adversary_image(image_arr, model)
    # embed()
