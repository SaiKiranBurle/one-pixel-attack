from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions


class Vgg16(object):
    def __init__(self):
        self.model = VGG16()

    def predict(self, images, top=100):
        images = preprocess_input(images)
        predictions = self.model.predict(images)
        labels = decode_predictions(predictions, top=top)
        return labels
