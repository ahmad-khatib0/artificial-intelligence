import os

from fastbook import *

from fastai.basics import Path
from fastai.vision.widgets import ImageClassifierCleaner, VBox


# from fastbook import (
#     DataBlock,
#     RandomSplitter,
#     download_url,
#     search_images_bing,
#     Path,
#     Image,
#     download_images,
#     get_image_files,
#     verify_image,
#     add_props,
#     GetAttr,
# )

key = os.getenv("AZURE_SEARCH_KEY", "XXX")

results = search_images_bing(key, "grizzly bear")
# ims = results.attrgot('content_url')
# len(ims)

ims = [
    "http://3.bp.blogspot.com/-S1scRCkI3vY/UHzV2kucsPI/AAAAAAAAA-k/YQ5UzHEm9Ss/s1600/Grizzly%2BBear%2BWildlife.jpg"
]

dest = "images/grizzly.jpg"
download_url(ims[0], dest)

im = Image.open(dest)

im.thumbnail((128.0, 128.0))

bear_types = "grizzly", "black", "teddy"
path = Path("bears")

if not path.exists():
    path.mkdir()
    for o in bear_types:
        dest = path / o
        dest.mkdir(exist_ok=True)
        results = search_images_bing(key, f"{o} bear")
        download_images(dest, urls=results.attrgot("content_url"))

fns = get_image_files(path)
print(fns)

failed = verify_image(fns)
print(failed)


# failed.map(Path.unlink) # run it if you saw failed images


# assemble the downloaded data in a format suitable for model training
# A DataLoader is a class that provides batches of a few items at a time to the GPU
bears = DataBlock(
    # specifying the types we want for the independent (independent variable is the thing we are using
    # to make predictions from) and dependent variables (our target (bear category))
    blocks=(ImageBlock, CategoryBlock),
    # our underlying items will be file paths. We have to tell fastai how to get a list of those files.
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    # gets the name of the folder a file is in, Because we put each of our bear images into
    # folders based on the type of bear, this is going to give us the labels that we need.
    get_y=parent_label,  # what function to call to create the labels in our dataset
    item_tfms=Resize(128),
)


# will make the photos don't appear as they are in real life
# bears = bears.new(item_tfms=Resize(128, ResizeMethod.Squish))
# fill in black, it is a wast of compute
# bears = bears.new(item_tfms=Resize(128, ResizeMethod.Pad, pad_mode='zeros'))
# bears = bears.new(item_tfms=RandomResizedCrop(128, min_scale=0.3))
# bears = bears.new(item_tfms=Resize(128), batch_tfms=aug_transforms(mult=2))

bears = bears.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5), batch_tfms=aug_transforms()
)

dls = bears.dataloaders(path)

# dls.valid.show_batch(max_n=4, nrows=1, unique=True) # for Pad, Squish
# dls.train.show_batch(max_n=8, nrows=2, unique=True) # for aug_transforms(mult=2)

learn = cnn_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)

# to see whether the mistakes the model is making are mainly thinking that grizzlies are teddies
# (that would be bad for safety!), or that grizzlies are black bears, or something else.
# To visualize this, we can create a confusion matrix:
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

# shows us the images with the highest loss in our dataset.
interp.plot_top_losses(5, nrows=1)

cleaner = ImageClassifierCleaner(learn)
cleaner
# ImageClassifierCleaner doesn’t do the deleting or changing of labels for you
for idx in cleaner.delete():
    cleaner.fns[idx].unlink()

# To move images for which we’ve selected a different category
for cat, idx in cleaner.change():
    shutil.move(str(cleaner.fns[idx]), path/cat)

# When you call export, fastai will save a file called export.pkl:
learn.export()

path = Path()
path.ls(file_exists='.pkl')  # => export.pkl:

# When we use a model for getting predictions, instead of training, we call it inference. To create
# our inference learner from the exported file, we use load_learner (in this case, this isn’t really
# necessary, since we already have a working Learner in our notebook; we’re doing it here so you can
# see the whole process end to end):
learn_inf = load_learner(path/'export.pkl')

# When we’re doing inference, we’re generally getting predictions for just one
# image at a time. To do this, pass a filename to predict:
learn_inf.predict('images/grizzly.png')

learn_inf.dls.vocab  # i.e  ['black','grizzly','teddy']


btn_upload = widgets.FileUpload()
btn_upload

img = PILImage.create(btn_upload.data[-1])

# We can use an Output widget to display it:
out_pl = widgets.Output()
out_pl.clear_output()
with out_pl:
    display(img.to_thumb(128, 128))
out_pl

pred, pred_idx, probs = learn_inf.predict()

lbl_pred = widgets.Label()
lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'
lbl_pred  # => E,g: 'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'

# a button to do the classification
btn_run = widgets.Button(description="Classify")
btn_run


def on_click_classify(change):
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl:
        display(img.to_thumb(128, 128))
    pred, pred_idx, probs = learn_inf.predict(img)
    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'


btn_run.on_click(on_click_classify)

# put them all in a vertical box (VBox) to complete our GUI
VBox([widgets.Label("Seler your Bear!"), btn_upload, btn_run, out_pl, lbl_pred])
