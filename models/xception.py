from keras.applications.xception import Xception, preprocess_input, decode_predictions


class Xception_wrapper(object):
    def __init__(self):
        self.model = Xception()
        self.name = "Xception"
        self.input_size = (224, 224, 3)
        self.num_classes = 1000

    def predict(self, images, top=None):
        if top is None:
            top = self.num_classes
        images = preprocess_input(images)
        predictions = self.model.predict(images)
        labels = decode_predictions(predictions, top=top)
        return labels
