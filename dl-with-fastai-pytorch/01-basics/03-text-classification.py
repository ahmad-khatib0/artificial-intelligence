from fastai.text.all import *


# train a model that can classify the sentiment of a movie review

dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid="test")

learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)

learn.fine_tune(4, 1e-2)
learn.predict("I really liked that movie!")

# ('pos', tensor(1), tensor([0.0041, 0.9959]))
# Here we can see the model has considered the review to be positive. The second part of the
# result is the index of “pos” in our data vocabulary, and the last part is the probabilities
# attributed to each class (99.6% for “pos” and 0.4% for “neg”).

# doc(learn.predict) # see document of method
