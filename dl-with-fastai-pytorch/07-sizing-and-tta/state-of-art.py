from fastai.vision.all import *

path = untar_data(URLs.IMAGENETTE)

dblock = DataBlock(
    blocs=(ImageBlock(), CategoryBlock()),
    get_items=get_image_files,
    get_y=parent_label,
    item_tfms=Resize(460),
    batch_tfms=aug_transforms(size=224, min_scale=0.75),
)

dls = dblock.dataloaders(path, bs=64)

# Then we’ll do a training run that will serve as a baseline:
model = xresnet50()
learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy)
learn.fit_one_cycle(5, 3e-3)


x, y = dls.one_batch()
x.mean(dim=[0, 2, 3]), x.std(dim=[0.2, 3])
# (TensorImage([0.4842, 0.4711, 0.4511], device='cuda:5'), TensorImage([0.2873, 0.2893, 0.3110], device='cuda:5'))


def get_dls(bs, size):
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        get_y=parent_label,
        item_tfms=Resize(460),
        batch_tfms=[*aug_transforms(size=size, min_scale=0.75),
                    Normalize.from_stats(*imagenet_stats)]
    )
    return dblock.dataloaders(path, bs=bs)


dls = get_dls(64, 224)
x, y = dls.one_batch()
x.mean(dim=[0, 2, 3]), x.std(dim=[0, 2, 3])
# (TensorImage([-0.0787, 0.0525, 0.2136], device='cuda:5'), TensorImage([1.2330, 1.2112, 1.3031], device='cuda:5'))


# Let’s check what effect this had on training our model:
model = xresnet50()
learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy)
learn.fit_one_cycle(5, 3e-3)


# Progressive Resizing

dls = get_dls(128, 128)
learn = Learner(dls, xresnet50(),
                loss_func=CrossEntropyLossFlat(), metrics=accuracy)
learn.fit_one_cycle(4, 3e-3)

# Then you can replace the DataLoaders inside the Learner, and fine-tune:
learn.dls = get_dls(64, 224)
learn.fine_tune(5, 1e-3)

# You can pass any DataLoader to fastai’s tta method; by default, it will use your validation set:
preds, targs = learn.tta()
accuracy(preds, targs).item()  # 0.8737863898277283

# Here is how we train a model with Mixup(like the intersecion of church and gas-station photo):
model = xresnet50()
learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(),
                metrics=accuracy, cbs=MixUp)
learn.fit_one_cycle(5, 3e-3)


# Label Smoothing
model = xresnet50()
learn = Learner(
    dls, model, loss_func=LabelSmoothingCrossEntropy(), metrics=accuracy)
learn.fit_one_cycle(5, 3e-3)
# As with Mixup, you won’t generally see significant improvements from label smooth‐
# ing until you train more epochs. Try it yourself and see: how many epochs do you
# have to train before label smoothing shows an improvement?
