import os
import torch
import torch.nn.functional as F
from torch.autograd import grad
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

print(torch.__version__)

print(torch.cuda.is_available())

tensor0d = torch.tensor(1)
tensor1d = torch.tensor([1, 2, 3])
tensor2d = torch.tensor([[1, 2], [3, 4]])
tensor3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print(tensor1d.dtype)  # torch.int64

floatvec = torch.tensor([1.0, 2.0, 3.0])
print(floatvec.dtype)  # torch.float32

floatvec = tensor1d.to(torch.float32)
print(floatvec.dtype)  # torch.float32

tensor2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(tensor2d.shape)  # torch.Size([2, 3])

# To reshape the tensor into a 3 × 2 tensor, we can use the .reshape
print(tensor2d.reshape(3, 2))  # tensor([[1, 2], [3, 4], [5, 6]])

# However, note that the more common command for reshaping tensors in PyTorch is .view():
print(tensor2d.view(3, 2))  # tensor([[1, 2], [3, 4], [5, 6]])


# Transpose a tensor, which means flipping it across its diagonal
print(tensor2d.T)  # tensor([[1, 4], [2, 5], [3, 6]])

#  the common way to multiply two matrices in PyTorch:
print(tensor2d.matmul(tensor2d.T))  # tensor([[14, 32], [32, 77]])
#  we can also adopt the @ operator, which accomplishes the same thing:
print(tensor2d @ tensor2d.T)  # tensor([[14, 32], [32, 77]])


# The following listing implements the forward pass (prediction step) of a simple logistic regression
# classifier, which can be seen as a single-layer neural network. It returns a score between 0 and 1,
# which is compared to the true class label (0 or 1) when computing the loss.

y = torch.tensor([1.0])  # true label
x1 = torch.tensor([1.1])  # input feature
w1 = torch.tensor([2.2])  # Weight parameter
b = torch.tensor([0.0])  # bias unit
z = x1 * w1 + b  # Net Input
a = torch.sigmoid(z)  # Activation and output

loss = F.binary_cross_entropy(a, y)


#  we can compute the gradient of the loss concerning the model parameter w1 ,
y = torch.tensor([1.0])
x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

z = x1 * w1 + b
a = torch.sigmoid(z)
loss = F.binary_cross_entropy(a, y)
# By default, PyTorch destroys the computation graph after calculating the gradients to
# free memory. However, since we will reuse this computation graph shortly, we set retain_graph=True
# so that it stays in memory.
grad_L_w1 = grad(loss, w1, retain_graph=True)
grad_L_b = grad(loss, b, retain_graph=True)

# The resulting values of the loss given the model’s parameters are
print(grad_L_w1), print(grad_L_b)
# (tensor([-0.0898]),),  (tensor([-0.0817]),)

# Here, we have been using the grad function manually, which can be useful for experimentation,
# debugging, and demonstrating concepts. But, in practice, PyTorch provides even more high-level
# tools to automate this process. For instance, we can call .backward on the loss, and PyTorch
# will compute the gradients of all the leaf nodes in the graph, (stored in grad attr)
loss.backward()

print(w1.grad), print(b.grad)
# (tensor([-0.0898]),), (tensor([-0.0817]),)


# classic multilayer perceptron with two hidden layers to illustrate a typical usage of the Module class.


class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            # Linear layer takes the number of input and output nodes
            torch.nn.Linear(num_inputs, 30),
            # Nonlinear activation functions are placed between the hidden layers:
            torch.nn.ReLU(),
            #
            # # 2nd hidden layer
            # The number of output nodes of one hidden layer has to match the
            # number of inputs of the next layer.
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            #
            # output layer
            torch.nn.Linear(20, num_inputs),
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits  # The outputs of the last layer are called logits.


model = NeuralNetwork(50, 3)
print(model)
# NeuralNetwork(
#     (layers): Sequential(
#         (0): Linear(in_features=50, out_features=30, bias=True)
#         (1): ReLU()
#         (2): Linear(in_features=30, out_features=20, bias=True)
#         (3): ReLU()
#         (4): Linear(in_features=20, out_features=3, bias=True)
#     )
# )


# let’s check the total number of trainable parameters of this model:
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad_)

print("Total number of trainable model parameters:", num_params)  # 2213
# Each parameter for which requires_grad=True counts as a
# trainable parameter and will be updated during training

print(model.layers[0].weight)
# tensor([
# [ 0.1174, -0.1350, -0.1227, ..., 0.0275, -0.0520, -0.0192],
# [-0.0169, 0.1265, 0.0255, ..., -0.1247, 0.1191, -0.0698],
# ...,
# [-0.0159, 0.0587, -0.0916, ..., -0.1153, 0.0700, 0.0770],
# [-0.1019, 0.1345, -0.0176, ..., 0.0114, -0.0559, -0.0088]],
# requires_grad=True)

# (Similarly, you could access the bias vector via model.layers[0].bias.)
print(model.layers[0].weight.shape)  # show its dimensions:
# torch.Size([30, 50])

# The weight matrix here is a 30 × 50 matrix, and we can see that requires_grad is set to True, which
# means its entries are trainable—this is the default setting for weights and biases in torch.nn.Linear.

# while we want to keep using small random numbers as initial values for
# our layer weights, we can make the random number initialization reproducible by
# seeding PyTorch’s random number generator via manual_seed:
torch.manual_seed(123)
model = NeuralNetwork(50, 3)
print(model.layers[0].weight)
# tensor([
#     [ -0.0577, 0.0047, -0.0702, ..., 0.0222, 0.1260, 0.0865],
#     [ 0.0502, 0.0307, 0.0333, ..., 0.0951, 0.1134, -0.0297],
#     [ 0.1077, -0.1108, 0.0122, ..., 0.0108, -0.1049, -0.1063],
#     ...,
#     [-0.0787, 0.1259, 0.0803, ..., 0.1218, 0.1303, -0.1351],
#     [ 0.1359, 0.0175, -0.0673, ..., 0.0674, 0.0676, 0.1058],
#     [ 0.0790, 0.1343, -0.0293, ..., 0.0344, -0.0971, -0.0509]],
# requires_grad=True)

torch.manual_seed(123)
X = torch.randn((1, 50))  # 50 elements
# Note that our network expects 50-dimensional feature vectors
# when we call model(x) it will automatically execute the forward pass of the model.
out = model(X)
print(X.dtype), print(X.shape)  # torch.float32,  torch.Size([1, 50])
print(out)
# tensor([[-0.1262, 0.1080, -0.1792]], grad_fn=<AddmmBackward0>) # returned 3 outputs (num_outputs)

# Here, grad_fn=<AddmmBackward0> represents the last-used function to compute a variable in the
# computational graph. In particular, grad_fn=<AddmmBackward0> means that the tensor we are
# inspecting was created via a matrix multiplication and addition operation. PyTorch will use this
# information when it computes gradients during back-propagation. The <AddmmBackward0> part of
# grad_fn=<AddmmBackward0> specifies the operation performed. In this case, it is an Addmm
# operation. Addmm stands for matrix multiplication ( mm) followed by an addition (Add).

with torch.no_grad():
    out = model(X)
print(out)
# tensor([[-0.1262, 0.1080, -0.1792]])
# If we just want to use a network without training or backpropagation—for example, if we use it
# for prediction after training—constructing this computational graph for backpropagation can be
# wasteful as it performs unnecessary computations and consumes additional memory. So, when we
# use a model for inference (for instance, making predictions) rather than training, the best
# practice is to use the torch.no_grad() context manager. This tells PyTorch that it doesn’t need
# to keep track of the gradients, which can result in significant savings in memory and computation


# If we want to compute class-membership probabilities for our predictions,
# we have to call the softmax function explicitly:
with torch.no_grad():
    out = torch.softmax(model(X), dim=1)
print(out)  # tensor([[0.3113, 0.3934, 0.2952]]))


# creating simple data loader:

# a simple toy dataset of five training examples with two features each.
X_train = torch.tensor(
    [[-1.2, 3.1], [-0.9, 2.9], [-0.5, 2.6], [2.3, -1.1], [2.7, -1.5]]
)
y_train = torch.tensor([0, 0, 0, 1, 1])

X_test = torch.tensor([[-0.8, 2.8], [2.6, -1.6]])

y_test = torch.tensor([0, 1])


class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    # Instructions for retrieving exactly one data record and the corresponding label
    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    # Instructions for returning the total length of the dataset
    def __len__(self):
        return self.labels.shape[0]


train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)

print(len(train_ds))  # 5

torch.manual_seed(123)
train_loader = DataLoader(
    # The ToyDataset instance created earlier serves as input to the data loader.
    dataset=train_ds,
    batch_size=2,
    shuffle=True,  # Whether or not to shuffle the data
    num_workers=0,  # The number of background processes
)


test_loader = DataLoader(
    dataset=test_ds,
    batch_size=2,
    shuffle=False,  # It is not necessary to shuffle a test dataset.
    num_workers=0,
)


for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx+1}:", x, y)
# Batch 1: tensor([[-1.2000,  3.1000], [-0.5000,  2.6000]]) tensor([0, 0])
# Batch 2: tensor([[ 2.3000, -1.1000], [-0.9000,  2.9000]]) tensor([1, 0])
# Batch 3: tensor([[ 2.7000, -1.5000]]) tensor([1])

# if you iterate over the dataset a second time, you will see that the shuffling order will
# change. This is desired to prevent deep neural networks from getting caught in repetitive
# update cycles during training

# In practice, having a substantially smaller batch as the last batch in a training epoch (note the
# batch number 3 in the output) can disturb the convergence during training. To prevent this, set
# drop_last=True, which will drop the last batch in each epoch,
train_loader = DataLoader(
    dataset=train_ds, batch_size=2, shuffle=True, num_workers=0, drop_last=True
)

for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx+1}:", x, y)
# output 2 batches


# A typical training loop
torch.manual_seed(123)
# The dataset has two features and two classes:
model = NeuralNetwork(num_inputs=2, num_outputs=2)
# The optimizer needs to know which parameters to optimize:
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

num_epochs = 5

for epoch in range(num_epochs):
    model.train()

    for batch_idx, (features, labels) in enumerate(train_loader):
        logits = model(features)

        loss = F.cross_entropy(logits, labels)
        # sets the gradients from the previous round to 0 to prevent unintended gradient accumulation
        optimizer.zero_grad()
        # compute the gradients of the loss given the model parameters
        loss.backward()
        # The optimizer uses the gradients to update the model parameters
        optimizer.step()
        print(
            f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
            f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
            f" | Train/Val Loss: {loss:.2f}"
        )

#
# Epoch: 001/005 | Batch 000/002 | Train/Val Loss: 0.75
# Epoch: 001/005 | Batch 001/002 | Train/Val Loss: 0.65
# Epoch: 002/005 | Batch 000/002 | Train/Val Loss: 0.44
# Epoch: 002/005 | Batch 001/002 | Train/Val Loss: 0.13
# Epoch: 003/005 | Batch 000/002 | Train/Val Loss: 0.03
# Epoch: 003/005 | Batch 001/002 | Train/Val Loss: 0.00
# Epoch: 004/005 | Batch 000/002 | Train/Val Loss: 0.02
# Epoch: 004/005 | Batch 001/002 | Train/Val Loss: 0.02
# Epoch: 005/005 | Batch 000/002 | Train/Val Loss: 0.00
# Epoch: 005/005 | Batch 001/002 | Train/Val Loss: 0.02
#
# as you can see the loss reaches 0 after 3 epochs

# After we have trained the model, we can use it to make predictions:
model.eval()
with torch.no_grad():
    outputs = model(X_train)

print(outputs)
# tensor([[ 2.8569, -4.1618], [ 2.5382, -3.7548], [ 2.0944, -3.1820], [-1.4814, 1.4816], [-1.7176, 1.7342]])

# To obtain the class membership probabilities, we can then use PyTorch’s softmax
torch.set_printoptions(sci_mode=False)  # make the outputs more legible
probas = torch.softmax(outputs, dim=1)
print(probas)

# tensor([
#     [ 0.9991, 0.0009], [ 0.9982, 0.0018],
#     [ 0.9949, 0.0051], [ 0.0491, 0.9509], [ 0.0307, 0.9693]
# ])
# Here, the first value (column) means that the training example has a 99.91% probability of belonging
# to class 0 and a 0.09% probability of belonging to class 1,

#
# We can convert these values into class label predictions using PyTorch’s argmax
# function, which returns the index position of the highest value in each row if we set
# dim=1 (setting dim=0 would return the highest value in each column instead):
predictions = torch.argmax(probas, dim=1)
print(predictions)  # tensor([0, 0, 0, 1, 1])

# Note that it is unnecessary to compute softmax probabilities to obtain the class labels.
# We could also apply the argmax function to the logits (outputs) directly:
predictions = torch.argmax(outputs, dim=1)
print(predictions)  # tensor([0, 0, 0, 1, 1])


# Since the training dataset is relatively small, we could compare it to the true training labels
predictions == y_train
# tensor([True, True, True, True, True])

# count the number of correct predictions:
torch.sum(predictions == y_train)  # 5 (so:  5/5 × 100% = 100% prediction accuracy.)


def compute_accuracy(model, dataloader):
    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)
        # Returns a tensor of True/False values depending on whether the labels match
        compare = labels == predictions
        # The sum operation counts the number of True values:
        correct += torch.sum(compare)
        total_examples += len(compare)

    # The fraction of correct prediction, a value between 0 and 1. .item()
    # returns the value of the tensor as a Python float.
    return (correct / total_examples).item()


print(compute_accuracy(model, train_loader))  # 1.0

print(compute_accuracy(model, test_loader))  # 1.0


# Save it so we can reuse it later. The model’s state_dict is a Python dictionary object that maps
# each layer in the model to its trainable parameters (weights and biases).
torch.save(model.state_dict(), "model.pth")

# The line model = NeuralNetwork(2, 2) is not strictly necessary if you execute this code in the
# same session where you saved a model. However, I included it here to illustrate that we need an
# instance of the model in memory to apply the saved parameters. Here, the NeuralNetwork(2, 2)
# architecture needs to match the original saved model exactly.
model = NeuralNetwork(2, 2)
model.load_state_dict(torch.load("model.pth"))


tensor_1 = torch.tensor([1.0, 2.0, 3.0])
tensor_2 = torch.tensor([4.0, 5.0, 6.0])

print(tensor_1 + tensor_2)  # tensor([5., 7., 9.])


# to transfer these tensors onto a GPU and perform the addition there:
tensor_1 = tensor_1.to("cuda")
tensor_2 = tensor_2.to("cuda")
print(tensor_1 + tensor_2)  # tensor([5., 7., 9.], device='cuda:0')


tensor_1 = tensor_1.to("cpu")
print(tensor_1 + tensor_2)
# this will fail,  where one tensor resides on the CPU and the other on the GPU


# run on a single GPU
torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2, num_outputs=2)

# Defines a device variable that defaults to a GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transfers the model onto the GPU
model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
num_epochs = 5


for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
        features, labels = (
            features.to(device),
            labels.to(device),
        )  # Transfers the data onto the GPU
        logits = model(features)
        loss = F.cross_entropy(logits, labels)  # loss_function

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### LOGGING
        print(
            f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
            f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
            f" | Train/Val Loss: {loss:.2f}"
        )

    model.eval()
    # Insert optional model evaluation code

# Epoch: 001/003 | Batch 000/002 | Train/Val Loss: 0.75
# Epoch: 001/003 | Batch 001/002 | Train/Val Loss: 0.65
# Epoch: 002/003 | Batch 000/002 | Train/Val Loss: 0.44
# Epoch: 002/003 | Batch 001/002 | Train/Val Loss: 0.13
# Epoch: 003/003 | Batch 000/002 | Train/Val Loss: 0.03
# Epoch: 003/003 | Batch 001/002 | Train/Val Loss: 0.00


# Training with multiple GPUs
def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"  # Address of the main node
    os.environ["MASTER_PORT"] = "12345"  # Any free port on the machine
    init_process_group(
        # nccl stands for NVIDIA Collective Communication Library.
        # The NCCL backend designed for GPU-to-GPU communication
        backend="ncl",
        rank=rank,  # rank refers to the index of GPU we want to use (process identifier)
        world_size=world_size,  # the number of gpus to use (total number of processes)
    )
    # Sets the current GPU device on which tensors will be allocated and operations will be performed
    torch.cuda.set_device(rank)


def prepare_dataset():
    # insert dataset preparation code
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=False,  # DistributedSampler takes care of the shuffling now
        pin_memory=True,  # Enables faster memory transfer when training on GPU
        drop_last=True,
        # Splits the dataset into distinct, non-overlapping subsets for each process (GPU)
        sampler=DistributedSampler(train_ds),
    )

    return train_loader, test_loader


def compute_accuracy_multi_gpu(model, dataloader, device):
    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(dataloader):
        with torch.no_grad():
            logits = model(features)

        predictions = torch.argmax(logits, dim=1)
        # Returns a tensor of True/False values depending on whether the labels match
        compare = labels == predictions
        # The sum operation counts the number of True values:
        correct += torch.sum(compare)
        total_examples += len(compare)

    # The fraction of correct prediction, a value between 0 and 1. .item()
    # returns the value of the tensor as a Python float.
    return (correct / total_examples).item()


def main(rank, world_size, num_epochs):
    "The main function running the model training"
    ddp_setup(rank, world_size)
    train_loader, test_loader = prepare_dataset()
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    model.to(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    # DDP enables the synchronization of the gradients between the different GPUs during training.
    model = DDP(model, device_ids=[rank])
    for epoch in range(num_epochs):
        for features, labels in train_loader:
            features, labels = features.to(rank), labels.to(rank)  # rank is the gpu id
            # insert model prediction and backpropagation code
            print(
                f"[GPU{rank}] Epoch: {epoch+1:03d}/{num_epochs:03d}"
                f" | Batchsize {labels.shape[0]:03d}"
                f" | Train/Val Loss: {loss:.2f}"
            )

    model.eval()

    train_acc = compute_accuracy_multi_gpu(model, train_loader, device=rank)
    print(f"[GPU{rank}] Test accuracy", train_acc)
    test_acc = compute_accuracy_multi_gpu(model, test_loader, device=rank)
    print(f"[GPU{rank}] Test accuracy", test_acc)
    destroy_process_group()  # cleanup resource allocation


# executed when we run the code as a Python script instead of importing it as a module
if __name__ == "__main__":
    print("Number of GPUs available:", torch.cuda.device_count())
    torch.manual_seed(123)
    num_epochs = 3
    world_size = torch.cuda.device_count()
    # Launches the main function using multiple processes, nprocs=world_size means one process per GPU.
    # Note that the main function has a rank argument that we don’t include in the mp.spawn() call.
    # That’s because the rank, which refers to the process ID we use as the GPU ID, is already passed
    # automatically.
    mp.spawn(main, args=(world_size, num_epochs), nprocs=world_size)


# single gpu output
# PyTorch version: 2.2.1+cu117
# CUDA available: True
# Number of GPUs available: 1
# [GPU0] Epoch: 001/003 | Batchsize 002 | Train/Val Loss: 0.62
# [GPU0] Epoch: 001/003 | Batchsize 002 | Train/Val Loss: 0.32
# [GPU0] Epoch: 002/003 | Batchsize 002 | Train/Val Loss: 0.11
# [GPU0] Epoch: 002/003 | Batchsize 002 | Train/Val Loss: 0.07
# [GPU0] Epoch: 003/003 | Batchsize 002 | Train/Val Loss: 0.02
# [GPU0] Epoch: 003/003 | Batchsize 002 | Train/Val Loss: 0.03
# [GPU0] Training accuracy 1.0
# [GPU0] Test accuracy 1.0

# a machine with two GPUs
# PyTorch version: 2.2.1+cu117
# CUDA available: True
# Number of GPUs available: 2
# [GPU1] Epoch: 001/003 | Batchsize 002 | Train/Val Loss: 0.60
# [GPU0] Epoch: 001/003 | Batchsize 002 | Train/Val Loss: 0.59
# [GPU0] Epoch: 002/003 | Batchsize 002 | Train/Val Loss: 0.16
# [GPU1] Epoch: 002/003 | Batchsize 002 | Train/Val Loss: 0.17
# [GPU0] Epoch: 003/003 | Batchsize 002 | Train/Val Loss: 0.05
# [GPU1] Epoch: 003/003 | Batchsize 002 | Train/Val Loss: 0.05
# [GPU1] Training accuracy 1.0
# [GPU0] Training accuracy 1.0
# [GPU1] Test accuracy 1.0
# [GPU0] Test accuracy 1.0


# If you wish to restrict the number of GPUs used for training on a multi-GPU machine,
# CUDA_VISIBLE_DEVICE=0 python some_script.py #  the GPU with index 0

# if your machine has four GPUs and you only want to use the first and third GPU
# CUDA_VISIBLE_DEVICE=0,2 python some_script.py #
