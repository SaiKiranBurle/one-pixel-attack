from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions


class Vgg16_wrapper(object):
    def __init__(self):
        self.model = VGG16()
        self.name = "VGG 16"
        self.input_size = (224, 224, 3)
        self.num_classes = 1000

    def predict(self, images, top=None):
        if top is None:
            top = self.num_classes
        images = preprocess_input(images)
        predictions = self.model.predict(images)
        labels = decode_predictions(predictions, top=top)
        return labels
