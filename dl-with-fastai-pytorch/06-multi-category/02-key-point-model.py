from fastai.vision.all import *
from fastbook import plot_function

# A key point model. A key point refers to a specific location represented in an image—in this case,
# we’ll use images of people and we’ll be looking for the center of the person’s face in each image. That
# means we’ll actually be predicting two values for each image: the row and column of the face center.

# Biwi Kinect Head Pose dataset: https://icu.ee.ethz.ch/research/datsets.html


path = untar_data(URLs.BIWI_HEAD_POSE)
path.ls()
# There are 24 directories numbered from 01 to 24
# (#50) [Path('13.obj'), ... ,Path('20.obj'),Path('17')...]

(path/'01').ls()
# (#1000) [Path('01/frame_00281_pose.txt'), ... ,Path('01/frame_00324_rgb.jpg')...]

# Inside the subdirectories, we have different frames. Each of them comes with an image (_rgb.jpg)
# and a pose file (_pose.txt). We can easily get all the image files recursively with get_image_files,
# and then write a function that converts an image file name to its associated pose file:
img_files = get_image_files(path)


def img2pose(x): return Path(f'{str(x)[:-7]}pose.txt')


img2pose(img_files[0])
# Path('13/frame_00349_pose.txt')

im = PILImage.create(img_files[0])
im.shape  # (480, 640)
im.to_thumb(160)


cal = np.genfromtxt(path/'01'/'rgb.cal', skip_footer=6)


def get_ctr(f):
    "Extract the head center point, This function returns the coordinates as a tensor of two items"
    ctr = np.genfromtxt(img2pose(f), skip_header=3)
    c1 = ctr[0] * cal[0][0]/ctr[2] + cal[0][2]
    c2 = ctr[1] * cal[1][1]/ctr[2] + cal[1][2]
    return tensor([c1, c2])


get_ctr(img_files[0])  # tensor([384.6370, 259.4787])

# The only other difference from the previous data block examples is that the second
# block is a PointBlock. This is necessary so that fastai knows that the labels represent
# coordinates; that way, it knows that when doing data augmentation, it should do the
# same augmentation to these coordinates as it does to the images:
biwi = DataBlock(
    blocks=(ImageBlock, PointBlock),
    get_items=get_image_files,
    get_y=get_ctr,
    splitter=FuncSplitter(lambda o: o.parent.name == '13'),
    batch_tfms=[*aug_transforms(size=(240, 320)),
                Normalize.from_stats(*imagenet_stats)]
)

dls = biwi.dataloaders(path)
dls.show_batch(max_n=9, figsize=(8, 6))

xb, yb = dls.one_batch()
xb.shape, yb.shape
# (torch.Size([64, 3, 240, 320]), torch.Size([64, 1, 2]))

# Here’s an example of one row from the dependent variable
yb[0]  # tensor([[0.0111, 0.1810]], device='cuda:5')

# (coordinates in fastai and PyTorch are always rescaled between –1 and +1):
learn = cnn_learner(dls, resnet18, y_range=(-1, 1))

# y_range is implemented in fastai using sigmoid_range, which is defined as follows:


def sigmoid_range(x, lo, hi): return torch.sigmoid(x) * (hi-lo) + lo


plot_function(partial(sigmoid_range, lo=-1, hi=1), min=-4, max=4)

# We didn’t specify a loss function, which means we’re getting whatever
# fastai chooses as the default. Let’s see what it picked for us:
dls.loss_func  # FlattenedLoss of MSELoss()
# This makes sense, since when coordinates are used as the dependent variable, most of the time
# we’re likely to be trying to predict something as close as possible; that’s basically what MSELoss

# We can pick a good learning rate with the learning rate finder:
learn.lr_find()

lr = 2e-2
learn.fit_one_cycle(5, lr)

learn.show_results(ds_idx=1, max_n=3, figsize=(6, 8))


# Conclusion
# our pretrained model was trained to do image classification,
# and we fine-tuned for image regression.
