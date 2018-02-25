# Modified version of https://github.com/Hyperparticle/one-pixel-attack-keras/blob/master/networks/resnet.py

import numpy as np
from keras.models import load_model


class ResNet(object):
    def __init__(self):
        self.name = 'Resnet'
        self.model_filename = 'models/weights/resnet.h5'

        self.num_classes = 10
        self.input_size = (32, 32, 3)
        self.batch_size = 128
        self.acc = 0.9231  # Precalculated result for cifar10
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        self.model = load_model(self.model_filename)
        self.param_count = self.model.count_params()
        print "Successfully loaded {}".format(self.name)

    def preprocess_input(self, imgs):
        if imgs.ndim < 4:
            imgs = np.array([imgs])
        imgs = imgs.astype('float32')
        mean = [125.307, 122.95, 113.865]
        std = [62.9932, 62.0887, 66.7048]
        for img in imgs:
            for i in range(3):
                img[:, :, i] = (img[:, :, i] - mean[i]) / std[i]
        return imgs

    def predict(self, images):
        processed = self.preprocess_input(images)
        probabilities = self.model.predict(processed, batch_size=self.batch_size)
        decoded_predictions = list()
        for im in probabilities:
            decoded_prediction = list()
            for i in range(im.shape[0]):
                decoded_prediction.append(('probability', self.class_names[i], im[i]))
            decoded_predictions.append(decoded_prediction)
        return decoded_predictions
