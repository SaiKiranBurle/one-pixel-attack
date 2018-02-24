from models.vgg16 import Vgg16


def get_model_from_name(model_name):
    if model_name.lower() == "vgg16":
        return Vgg16()
    else:
        raise NotImplementedError
