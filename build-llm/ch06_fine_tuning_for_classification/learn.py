import torch

# linspace() returns a one-dimensional tensor of steps
# equally spaced points between start and end:
a = torch.linspace(3, 10, 5)
print("a = ", a)

b = torch.linspace(start=-10, end=10, steps=5)
print("b = ", b)

# a =  tensor([ 3.0000,  4.7500,  6.5000,  8.2500, 10.0000])
# b =  tensor([-10.,  -5.,   0.,   5.,  10.])
