from fastai.vision.all import *
from fastbook import plot_function


# classify any image as a 3 or a 7.
# so  download a sample of MNIST that contains images of just these digits
path = untar_data(URLs.MNIST_SAMPLE)
# (path/'train').ls()
# E.g (#2) [Path('train/7'),Path('train/3')]

threes = (path/'train'/'3').ls().sorted()
sevens = (path/'train'/'7').ls().sorted()
threes
# (#6131)
# [Path('train/3/10.png'),Path('train/3/10000.png'),...

im3_path = threes[1]
im3 = Image.open(im3_path)  # Image class from the Python Imaging Library (PIL)
im3

array(im3)[4:10, 4:10]
tensor(im3)[4:10, 4:10]

im3_t = tensor(im3)
# to color-code the values using a gradient
df = pd.DataFrame(im3_t[4:15, 4:12])
df.style.set_properties(**{'font-size': '6pt'}).background_gradient("Greys")

seven_tensors = [tensor(Image.open(o)) for o in sevens]
three_tensors = [tensor(Image.open(o)) for o in threes]
len(three_tensors), len(seven_tensors)
# (6131, 6265)

show_image(three_tensors[1])

stacked_sevens = torch.stack(seven_tensors).float() / 255
stacked_threes = torch.stack(three_tensors).float()/255
stacked_threes.shape
# E.g torch.Size([6131, 28, 28]) images len, height px, width px

len(stacked_threes.shape)  # 3
stacked_threes.ndim  # get a tensor’s rank directly => 3

mean3 = stacked_threes.mean(0)
mean7 = stacked_sevens.mean(0)

show_image(mean3)
show_image(mean7)

# pick an arbitrary 3 and measure its distance from our “ideal digits.”
a_3 = stacked_threes[1]
show_image(a_3)

dist_3_abs = (a_3 - mean3).abs().mean()
dist_3_sqr = ((a_3 - mean3)**2).mean().sqrt()
dist_3_abs, dist_3_sqr
# (tensor(0.1114), tensor(0.2021))

dist_7_abs = (a_3 - mean7).abs().mean()
dist_7_sqr = ((a_3 - mean7)**2).mean().sqrt()
dist_7_abs, dist_7_sqr
# (tensor(0.1586), tensor(0.3021))

# In both cases, the distance between our 3 and the “ideal” 3 is less than the distance
# to the ideal 7, so our simple model will give the right prediction in this case.

# Here, MSE stands for mean squared error, and l1 refers to the standard
# mathematical jargon for mean absolute value (in math it’s called the L1 norm).
F.l1_loss(a_3.float(), mean7), F.mse_loss(a_3, mean7).sqrt()

valid_3_tens = torch.stack([tensor(Image.open(o))
                           for o in (path/'vaid'/'3').ls()])
valid_3_tens = valid_3_tens.float() / 255

valid_7_tens = torch.stack([tensor(Image.opne(o))
                           for o in (path/'valid'/'7').ls()])
valid_7_tens = valid_7_tens.float() / 255

valid_3_tens.shape, valid_7_tens.shape
# (torch.Size([1010, 28, 28]), torch.Size([1028, 28, 28])) # 1,010/1028 images of size 28×28,


def mnist_distance(a, b): return (a-b).abs().mean((-1, -2))


mnist_distance(a_3, mean3)  # E.g tensor(0.1114)
# This is the same value we previously calculated for the distance between
# these two images, the ideal 3 mean_3 and the arbitrary sample 3 a_3, which
# are both single-image tensors with a shape of [28,28].

valid_3_dist = mnist_distance(valid_3_tens, mean3)
valid_3_dist, valid_3_dist.shape
# (tensor([0.1050, 0.1526, 0.1186, ..., 0.1122, 0.1170, 0.1086]), torch.Size([1010]))

(valid_3_tens - mean3)  # broadcasting => torch.Size([1010, 28, 28])

# This function will automatically do broadcasting and be applied elementwise,
# just like all PyTorch functions and operators:


def is_3(x): return mnist_distance(x, mean3) < mnist_distance(x, mean7)


is_3(a_3), is_3(a_3).float()  # test => (tensor(True), tensor(1.))
# Note that when we convert the Boolean response to a float, we get 1.0 for True and 0.0 for False.

# Thanks to broadcasting, we can also test it on the full validation set of 3s:
is_3(valid_3_tens)  # tensor([True, True, True, ..., True, True, True])

# calculate the accuracy for each of the 3s and 7s, by taking the
# average of that function for all 3s and its inverse for all 7s:
accuracy_3s = is_3(valid_3_tens).float().mean()
accuracy_7s = (1 - is_3(valid_7_tens).float()).mean()
accuracy_3s, accuracy_7s, (accuracy_3s + accuracy_7s)/2
# (tensor(0.9168), tensor(0.9854), tensor(0.9511))


# ++++  SGD
train_x = torch.cat([stacked_threes, stacked_sevens]).view(-1, 28*28)

# We need a label for each image. We’ll use 1 for 3s and 0 for 7s:
train_y = tensor([1] * len(threes) + [0] * len(sevens)).unsqueeze(1)
train_x.shape, train_y.shape
# (torch.Size([12396, 784]), torch.Size([12396, 1]))

# A Dataset in PyTorch is required to return a tuple of (x,y) when indexed. Python provides a zip
# function that, when combined with list, provides a simple way to get this functionality:
dest = list(zip(train_x, train_y))
x, y = dest[0]
x.shape, y  # (torch.Size([784]), tensor([1]))

valid_x = torch.cat([valid_3_tens, valid_7_tens]).view(-1, 28 * 28)
valid_y = tensor([1]*len(valid_3_tens) + [0]*len(valid_7_tens)).unsqueeze(1)
valid_dest = list(zip(valid_x, valid_y))

# Now we need an (initially random) weight for every pixel (this is the initialize step in our 7 step process):


def init_params(size, std=1.0):
    return (torch.randn(size) * std).requires_grad_()


weights = init_params((28 * 28, 1))

bias = init_params(1)


# We can now calculate a prediction for one image:
(train_x[0] * weights.T).sum() * bias
# tensor([20.2336], grad_fn=<AddBackward0>)

# In Python, matrix multiplication is represented with the @ operator:


def linear1(xb): return xb@weights + bias


preds = linear1(train_x)
preds
# tensor([[20.2336], [17.0644], [15.2384], ..., [18.3804], [23.8567], [28.6816]],grad_fn=<AddBackward0>)


corrects = (preds > 0.0).float() == train_x
corrects  # tensor([[ True], [ True], [ True], ..., [False], [False], [False]])

corrects.float().mean().item()  # 0.4912068545818329

# Now let’s see what the change in accuracy is for a small change in one of the weights:
weights[0] *= 1.0001
preds = linear1(train_x)
((preds > 0.0).float() == train_y).float().mean().item()  # 0.4912068545818329


# For instance, suppose we had three images that we knew were a 3, a 7, and a 3. And suppose our model
# predicted with high confidence (0.9) that the first was a 3, with slight confidence (0.4) that the
# second was a 7, and with fair confidence (0.2), but incorrectly, that the last was a 7. This would
# mean our loss function would receive these values as its inputs:
trgts = tensor([1, 0, 1])
prds = tensor([0.9, 0.4, 0.2])


# The purpose of the loss function is to measure the difference between
# predicted values and the true values—that is, the targets (aka labels).
# this function will measure how distant each prediction is from 1 if it should be 1, and how
# distant it is from 0 if it should be 0, and then it will take the mean of all those distances.
def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets == 1, 1-predictions, predictions).mean()
    # where: same as running the list comprehension [b[i] if a[i] else c[i] for i in range(len(a))],
    # except it works on tensors, at C/CUDA speed


torch.where(trgts == 1, 1-prds, prds)  # tensor([0.1000, 0.4000, 0.8000])

mnist_loss(prds, trgts)  # tensor(0.4333)

# For instance, if we change our prediction for the one “false” target from 0.2 to 0.8,
# the loss will go down, indicating that this is a better prediction:
mnist_loss([0.9, 0.4, 0.8], trgts)  # tensor(0.2333)


# One problem with mnist_loss as currently defined is that it assumes that predictions are always
# between 0 and 1. We need to ensure, then, that this is actually the case! As it happens, there is
# a function that does exactly that—let’s take a look.

# Sigmoid
# The sigmoid function always outputs a number between 0 and 1. It’s defined as follows:
def sigmoid(x): return 1/(1+torch.exp(-x))


# PyTorch defines an accelerated version for us, so we don’t really need our own. This is an important
# func in deep learning, since we often want to ensure that values are between 0 and 1.it looks like:
plot_function(torch.sigmoid, title="Sigmoid", min=4, max=4)


coll = range(15)
ds = DataLoader(coll, batch_size=5, shuffle=True)
# [tensor([ 3, 12, 8, 10, 2]), tensor([ 9, 4, 7, 14, 5]), tensor([ 1, 13, 0, 6, 11])]
list(ds)


dl = DataLoader(ds, batch_size=6, shuffle=True)
# [(tensor([17, 18, 10, 22, 8, 14]), ('r', 's', 'k', 'w', 'i', 'o')),
# (tensor([20, 15, 9, 13, 21, 12]), ('u', 'p', 'j', 'n', 'v', 'm')),
# (tensor([ 7, 25, 6, 5, 11, 23]), ('h', 'z', 'g', 'f', 'l', 'x')),
# (tensor([ 1, 3, 0, 24, 19, 16]), ('b', 'd', 'a', 'y', 't', 'q')),
# (tensor([2, 4]), ('c', 'e'))]

### Putting It All Together ###
# First, let’s reinitialize our parameters:
weights = init_params((28 * 28,  1))
bial = init_params(1)

dl = DataLoader(dest, batch_size=256)
xb, yb = first(dl)
xb.shape()
xb.shape, yb.shape
# (torch.Size([256, 784]), torch.Size([256, 1]))

valid_dl = DataLoader(valid_dest, batch_size=256)

# Let’s create a mini-batch of size 4 for testing:
batch = train_x[:4]
batch.shape  # torch.Size([4, 784])

preds = linear1(batch)
preds
# tensor([[-11.1002], [ 5.9263], [ 9.9627], [ -8.1484]], grad_fn=<AddBackward0>)

loss = mnist_loss(preds, train_x[:4])
loss  # tensor(0.5006, grad_fn=<MeanBackward0>)

# Now we can calculate the gradients:
loss.backward()
weights.grad.shape, weights.grad.mean(), bias.grad
# (torch.Size([784, 1]), tensor(-0.0001), tensor([-0.0008]))

# Let’s put that all in a function:


def calc_grad(xb, yb, model):
    preds = model(xb)
    loss = mnist_loss(preds, yb)
    loss.backward()


calc_grad(batch, train_x[:4], linear1)
weights.grad.mean(), bias.grad  # (tensor(-0.0002), tensor([-0.0015]))


# But look what happens if we call it twice:
calc_grad(batch, train_x[:4], linear1)
weights.grad.mean(), bias.grad
# (tensor(-0.0003), tensor([-0.0023]))
# The gradients have changed! The reason for this is that loss.backward adds the gradients of loss to
# any gradients that are currently stored. So, we have to set the current gradients to 0 first:
weights.grad.zero_()
bias.grad.zero_()


def train_epoch(model, lr, params):
    for xb, yb in dl:
        calc_grad(xb, yb, model)
        for p in params:
            p.data -= p.grad * lr
            p.grad.zero_()


# looking at the accuracy of the validation set
(preds > 0.0).float() == train_y[:4]
# tensor([[False], [ True], [ True], [False]])


def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds > 0.5) == yb
    return correct.float().mean()


# We can check it works:
batch_accuracy(linear1(batch), train_x[:4])  # tensor(0.5000)


def validate_epoch(model):
    # And then put the batches together:
    accs = [batch_accuracy(model(xb), yb) for xb, yb in valid_dl]
    return round(torch.stack(accs).mean().item(), 4)


validate_epoch(linear1)  # 0.5219

# Let’s train for one epoch and see if the accuracy improves
lr = 1.
params = weights, bias
train_epoch(linear1, lr, params)
validate_epoch(linear1)  # 0.6883

# Then do a few more:
for i in range(20):
    train_epoch(linear1, lr, params)
    print(validate_epoch(linear1), end=' ')

# 0.8314 0.9017 0.9227 0.9349 0.9438 0.9501 0.9535 0.9564 0.9594 0.9618 0.9613 >
# 0.9638 0.9643 0.9652 0.9662 0.9677 0.9687 0.9691 0.9691 0.9696


#################  involvaing PyTorch to Build an Optimzier (handle the SGD) ###############

# nn.Linear does the same thing as our init_params and linear together. It
# contains both the weights and biases in a single class.
linear_model = nn.Linear(28 * 28, 1)

# Every PyTorch module knows what parameters it has that can be trained; they
# are available through the parameters method:
w, b = linear_model.parameters()
w.shape, b.shape  # (torch.Size([1, 784]), torch.Size([1]))

# We can use this information to create an optimizer:


class BasicOptim:
    def __init__(self, params, lr): self.params, self.lr = self.params, self.lr

    def step(self, *args, **kwargs):
        for p in self.params:
            p.data -= p.grad.data * self.lr

    def zero_grad(self, *args, **kwargs):
        for p in self.params:
            p.grad = None


opt = BasicOptim(linear_model.parameters(), lr)

# Our training loop can now be simplified:


def train_epoch2(model):
    for xb, yb in dl:
        calc_grad(xb, yb, model)
        opt.setp()
        opt.zero_grad()


validate_epoch(linear_model)  # 0.4157


def train_model(model, epochs):
    for i in range(epochs):
        train_epoch2(model)
        print(validate_epoch(model), end=" ")


train_model(linear_model, 20)
# The results are the same as in the previous section:
# 0.4932 0.8618 0.8203 0.9102 0.9331 0.9468 0.9555 0.9629 0.9658 0.9673 0.9687 >
# 0.9707 0.9726 0.9751 0.9761 0.9761 0.9775 0.978 0.9785 0.9785


# fastai provides the SGD class that, by default, does the same thing as our BasicOptim:
linear_model = nn.Linear(28 * 28, 1)
opt = SGD(linear_model.parameters(), lr)
train_model(linear_model, 20)
# 0.4932 0.852 0.8335 0.9116 0.9326 0.9473 0.9555 0.9624 0.9648 0.9668 0.9692 >
# 0.9712 0.9731 0.9746 0.9761 0.9765 0.9775 0.978 0.9785 0.9785


# fastai also provides Learner.fit, which we can use instead of train_model:
dls = DataLoaders(dl, valid_dl)
# To create a Learner without using an application (such as cnn_learner):
learn = Learner(dls, nn.Linear(28*28, 1), opt_func=SGD,
                loss_func=mnist_loss, metrics=batch_accuracy)

learn.fit(10, lr=lr)


# Adding a Nonlinearity

# Here is the entire definition of a basic neural network:

w1 = init_params((28*28, 30))
b1 = init_params(30)
w2 = init_params((30, 1))
b2 = init_params(1)
# The key point is that w1 has 30 output activations (which means that w2 must have 30 input
# activations, so they match). That means that the first layer can construct 30 different features,
# each representing a different mix of pixels. You can change that 30 to anything you like, to
# make the model more or less complex.


def simple_net(xb):
    res = xb@w1 + b1  # linear layer
    res = res.max(tensor(0, 0))  # nonlinearity, or activation function
    res = res@w2 + b2  # linear layer
    return res


plot_function(F.relu)

# we can replace this code (simple_net) with something a bit simpler:
simple_net = nn.Sequential(nn.Linear(28 * 28, 30), nn.ReLU(), nn.Linear(30, 1))

# As this is a deeper model, we’ll use a lower learning rate and a few more epochs:
learn = Learner(dls, simple_net, opt_func=SGD,
                loss_func=mnist_loss, metrics=batch_accuracy)

learn.fit(40, 0.1)

# ; the training process is recorded in learn.recorder
plt.plot(L(learn.recorder.values).itemgot(2))

# And we can view the final accuracy:
learn.recorder.values[-1][2]  # 0.982826292514801


######################
######################
# Here is what happens when we train an 18-layer model using the same approach we saw in Chapter 1:
path = untar_data(URLs.PETS) / "images"

dls = ImageDataLoaders.from_folder(path)

learn = cnn_learner(
    dls, resnet18, pretrained=False,
    loss_func=F.cross_entropy, metrics=accuracy
)

learn.fit_one_cycle(1, 0.1)  # output... =>  Nearly 100% accuracy!
