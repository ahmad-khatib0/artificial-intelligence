from fastai.vision.all import *

# ruff: noqa: F405
path = untar_data(URLs.PASCAL_2007)

df = pd.read_csv(path/'train.csv')
df.head()

dblock = DataBlock()
# We can create a Datasets object from this. The only thing needed is a source:
dsets = dblock.datasets(df)

# This contains a train and a valid dataset, which we can index into:
dsets.train[0]  # returns a row of the DataFrame, twice, so:


def get_x(r): return path/'train'/r['fname']
def get_y(r): return r['labels'].split(' ')


# dblock = DataBlock(get_x=lambda r: r['fname'], get_y=lambda r: r['labels']) # same:
dblock = DataBlock(get_x=get_x, get_y=get_y)
dsets = dblock.datasets(df)
dsets.train[0]
# (Path('/home/sgugger/.fastai/data/pascal_2007/train/008663.jpg'), ['car', 'person'])

dblock = DataBlock(
    blocks=(ImageBlock, MultiCategoryBlock),
    get_x=get_x, get_y=get_y,
)
dsets = dblock.datasets(df)
dsets.train[0]
# (PILImage mode=RGB size=500x375, TensorMultiCategory([0., 0., ..., 0., 0.]))

# Let’s check what the categories (one-hot encoding) represent for this example:
idxs = torch.where(dsets.train[0][1] == 1.)[0]
dsets.train.vocab[idxs]  # (#1) ['dog']


def splitter(df):
    train = df.index[~df['is_valid']].tolist()
    valid = df.index[df['is_valid']].tolist()
    return train, valid


dblock = DataBlock(
    blocks=(ImageBlock, MultiCategoryBlock),
    splitter=splitter,
    get_x=get_x,
    get_y=get_y,
)
dsets = dblock.datasets(df)
dsets.train[0]
# (PILImage mode=RGB size=500x333, TensorMultiCategory([0., ..., 0., 1., 0., ... 0., 0.]))

dblock = DataBlock(
    blocks=(ImageBlock, MultiCategoryBlock),
    splitter=splitter,
    get_x=get_x,
    get_y=get_y,
    item_tfms=RandomResizedCrop(128, min_scale=0.35)
)
dls = dblock.dataloaders(df)
dls.show_batch(nrows=1, ncols=3)

learn = cnn_learner(dls, resnet18)

# grabbing a mini-batch from our DataLoader and then passing it to the model:
x, y = dls.train.one_batch()
activs = learn.model(x)
# torch.Size([64, 20]): batch size of 64, the probability of each of 20 categories
activs.shape

activs[0]
# tensor([
#    2.0258, -1.3543, 1.4640, 1.7754, -1.2820, -5.8053, 3.6130, 0.7193, -4.3683, -2.5001,
#   -2.8373, -1.8037, 2.0122, 0.6189, 1.9729, 0.8999, -2.6769, -0.3829, 1.2212, 1.6073],
#   device='cuda:0', grad_fn=<SelectBackward>
# )


def binary_cross_entropy(inputs, targets):
    inputs = inputs.sigmoid()
    return -torch.where(targets == 1, inputs, 1-inputs).log().mean()


loss_func = nn.BCEWithLogitsLoss()
loss = loss_func(activs, y)
# tensor(1.0082, device='cuda:5', grad_fn=<BinaryCrossEntropyWithLogitsBackward>)
loss


def accuracy_multi(inp, targ, thresh=0.5, sigmoid=True):
    "Compute accuracy when `inp` and `targ` are the same size."
    if sigmoid:
        inp = inp.sigmoid()
    return ((inp > thresh) == targ.bool()).float().mean()


learn = cnn_learner(dls, resnet50, metrics=partial(accuracy_multi, thresh=0.2))
learn.fine_tune(3, base_lr=3e-3, freeze_epochs=4)


learn.metrics = partial(accuracy_multi, thresh=0.1)
learn.validate()
# (#2) [0.10436797887086868,0.93057781457901] : returns the validation loss and metrics,

# If you pick a threshold that’s too high, you’ll be selecting only the
# objects about which the model is very confident:
learn.metrics = partial(accuracy_multi, thresh=0.99)
learn.validate()  # (#2) [0.10436797887086868,0.9416930675506592]

# We can find the best threshold by trying a few levels and seeing what
# works best. This is much faster if we grab the predictions just once:
preds, targs = learn.get_preds()

accuracy_multi(preds, targs, thresh=0.9, sigmoid=False)
# TensorMultiCategory(0.9554)

# We can now use this approach to find the best threshold level:
xs = torch.linspace(0.05, 0.95, 29)
accs = [accuracy_multi(preds, targs, thresh=i, sigmoid=False) for i in xs]
plt.plot(xs, accs)
