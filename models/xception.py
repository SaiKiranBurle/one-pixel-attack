from keras.applications.xception import Xception, preprocess_input, decode_predictions


class Xception_wrapper(object):
    def __init__(self):
        self.model = Xception()

    def predict(self, images, top=100):
        images = preprocess_input(images)
        predictions = self.model.predict(images)
        labels = decode_predictions(predictions, top=top)
        return labels
