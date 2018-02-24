from keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions


class MobileNet_wrapper(object):
    def __init__(self):
        self.model = MobileNet()

    def predict(self, images, top=100):
        images = preprocess_input(images)
        predictions = self.model.predict(images)
        labels = decode_predictions(predictions, top=top)
        return labels
