from fastai.collab import *
from fastai.tabular.all import URLs, untar_data

path = untar_data(URLs.ML_SAMPLE)

dls = CollabDataLoaders.from_csv(path / "ratings.csv")

# This model is predicting movie ratings on a scale of 0.5 to 5.0 to within around 0.6 average
# error. Since we’re predicting a continuous number, rather than a category, we have to tell
# fastai what range our target has, using the y_range parameter.
learn = collab_learner(dls, y_range=(0.5, 5.5))

# Although we’re not actually using a pretrained model (for the same reason that we didn’t for
# the tabular model), this example shows that fastai lets us use fine_tune anyway in this case
learn.fine_tune(10)

learn.show_results()
