from fastbook import *
# from fastai.vision.all import CategoryBlock, DataBlock, ImageBlock, RandomSplitter, RegexLabeller, Resize, URLs, aug_transforms, get_image_files, re, untar_data, using_attr
from fastai.vision.all import *
from fastai2.callback.fp16 import *
from torch import exp

path = untar_data(URLs.PETS)

path.ls()
# (#3) [Path('annotations'),Path('images'),Path('models')]

(path/"images").ls()

fname = (path/"images").ls()[0]

re.findall(r'(.+)_\d+.jpg$', fname.name)  # ['great_pyrenees']

pets = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(seed=42),
    get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name'),
    item_tfms=Resize(460),
    batch_tfms=aug_transforms(size=224, min_scale=0.75),
)

dls = pets.dataloaders(path/"images")
# Checking and Debugging a DataBlock (check labels, object positions (augmentation))
dls.show_patch(nrows=1, ncols=3)

# Critical Step here ATP: Debugging using summary (intencial error: removed the Resize)
pets1 = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(seed=42),
    get_y=using_attr(RegexLabeller(r'(.+)_\d+.jpg$'), 'name')
)

pets1.summary(path/'images')

# next step should be train a simple model (as a test in earlly)
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(2)


# Viewing Activations and Labels
# returns the dependent (y) and independent variables(x):
x, y = dls.one_batch()

preds, _ = learn.get_preds(dl=[(x, y)])
preds[0]

# The actual predictions are 37 probabilities between 0 and 1, which add up to 1 in total:
len(preds[0]), preds[0].sum()  # (37, tensor(1.0000))

acts = torch.randn((6, 2)) * 2
acts
# tensor([
#     [ 0.6734, 0.2576], [ 0.4689, 0.4607], [-2.2457, -0.3727],
#     [ 4.4164, -1.2760], [ 0.9233, 0.5347], [ 1.0698, 1.6187]]
# )

# We can’t just take the sigmoid of this directly, since we don’t get rows that add to 1
acts.sigmoid()

# use sigmoid directly on the two-activation version of our neural net.
(acts[:, 0]-acts[:, 1]).sigmoid()
# tensor([0.6025, 0.5021, 0.1332, 0.9966, 0.5959, 0.3661])


def softmax(x):
    """ we need a way to do all this that also works for more than two columns(being 3 or 7) 
        It turns out that this function, called softmax, is exactly that: """
    return exp(x) / exp(x).sum(dim=1, keepdim=True)


# use a confusion matrix to see where our model is doing well and where it’s doing badly (Model Interpretation):
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(12, 12), dpi=60)

# shows us the cells of the confusion matrix with the most incorrect predictions (at least 5 or more):
interp.most_confused(min_val=5)


# The Learning Rate Finder
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1, base_lr=0.1)  # test making learning rate really high
lr_min, lr_steep = learn.lr_find()

print(f"Minimum/10: {lr_min:.2e}, steepest point: {lr_steep:.2e}")
# Minimum/10: 8.32e-03, steepest point: 6.31e-03

learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(2, base_lr=3e-3)


# Unfreezing and Transfer Learning:
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fit_one_cycle(3, 3e-3)

# Then we’ll unfreeze the model:
learn.unfreeze()

learn.lr_find()
# (1.0964782268274575e-05, 1.5848931980144698e-06)

# Let’s train at a suitable learning rate:
learn.fit_one_cycle(6, lr_max=1e-5)


# Discriminative Learning Rates
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fit_one_cycle(3, 3e-3)
learn.unfreeze()
learn.fit_one_cycle(12, lr_max=slice(1e-6, 1e-4))

# fastai can show us a graph of the training and validation loss:
learn.recorder.plot_loss()


# tensor cores
learn = cnn_learner(dls, resnet50, metrics=error_rate).to_fp16()
learn.fine_tune(6, freeze_epochs=3)
