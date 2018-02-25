from keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions


class MobileNet_wrapper(object):
    def __init__(self):
        self.model = MobileNet()
        self.name = "MobileNet"
        self.input_size = (224, 224, 3)
        self.num_classes = 1000

    def predict(self, images, top=None):
        if top is None:
            top = self.num_classes
        images = preprocess_input(images)
        predictions = self.model.predict(images)
        labels = decode_predictions(predictions, top=top)
        return labels
