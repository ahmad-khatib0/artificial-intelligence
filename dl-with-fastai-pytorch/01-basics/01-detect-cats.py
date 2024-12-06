# import socket, warnings
from fastai.vision.all import *

path = untar_data(URLs.PETS) / "images"


# The filenames start with an uppercase letter if the image is a cat, and a lowercase letter otherwise.
def is_cat(x):
    return x[0].isupper()


dls = ImageDataLoaders.from_name_func(
    path,
    get_image_files(path),
    valid_pct=0.2,  # hold out 20%(0.2) for validation set,
    seed=42,  # random seed
    label_func=is_cat,  # how to get the labels from dataset
    item_tfms=Resize(224),  # each item is resized to a 224px,
)

# convolutional neural network (CNN) and specifies what architecture to use (i.e., what
# kind of model to create), what data we want to train it on, and what metric to use:
learn = cnn_learner(dls, resnet34, metrics=error_rate)

learn.fine_tune(1)
