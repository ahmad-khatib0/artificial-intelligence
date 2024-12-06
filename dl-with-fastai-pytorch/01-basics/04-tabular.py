from fastai.tabular.all import *

# This model is using the Adult dataset from the paper “Scaling Up the Accuracy of
# Naive-Bayes Classifiers: a Decision-Tree Hybrid” by Ron Kohavi,

path = untar_data(URLs.ADULT_SAMPLE)

# we had to tell fastai which columns are categorical (contain values that are one
# of a discrete set of choices, such as occupation) versus continuous (contain a number
# that represents a quantity, such as age).
dls = TabularDataLoaders.from_csv(
    path / "adult.csv",
    path=path,
    y_names="salary",
    cat_names=[
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
    ],
    cont_names=["age", "fnlwgt", "education-num"],
    procs=[Categorify, FillMissing, Normalize],
)

learn = tabular_learner(dls, metrics=accuracy)

# There is no pretrained model available for this task (in general, pretrained models are not
# widely available for any tabular modeling tasks) so we don’t use fine_tune in this case.
# Instead, we use fit_one_cycle, the most commonly used method for training fastai models
# from scratch (i.e., without transfer learning):
learn.fit_one_cycle(3)
