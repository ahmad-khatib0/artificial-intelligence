from fastai.vision.all import *

# For instance, let’s talk about something that is critically important for autonomous vehicles:
# localizing objects in a picture. If a self-driving car doesn’t know where a pedestrian is,
# then it doesn’t know how to avoid one! Creating a model that can recognize the content of every
# individual pixel in an image is called segmentation. Here is how we can train a
# segmentation model with fastai, using a subset of the CamVid dataset from the paper
# “Semantic Object Classes in Video: A High-Definition Ground Truth Database” by Gabriel J. Brostow et al.:

path = untar_data(URLs.CAMVID_TINY)
dls = SegmentationDataLoaders.from_label_func(
    path,
    bs=8,
    fnames=get_image_files(path / "images"),
    label_func=lambda o: path / "labels" / f"{o.stem}_P{o.suffix}",
    codes=np.loadtxt(path / "codes.txt", dtype=str),
)

learn = unet_learner(dls, resnet34)
learn.fine_tune(8)

learn.show_results(max_n=6, figsize=(7,8))
