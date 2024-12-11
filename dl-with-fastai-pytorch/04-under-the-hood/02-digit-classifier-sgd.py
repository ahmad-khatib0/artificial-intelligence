from fastai.vision.all import *
from fastbook import plot_function

# define a very simple function, the quadratic—let’s pretend that this is our loss function,
# and x is a weight parameter of the function:


def f(x): return x**2


plot_function(f, 'x', 'x**2')

# The sequence of steps we described earlier (info file: gradient descent process) starts by:

# picking a random value for a parameter, and calculating the value of the loss:
plt.scatter(-1.5, f(1.5), color="red")


#  pick a tensor value at which we want gradients
xt = tensor(3.).requires_grad_()
yt = f(xt)  # calculate our function with that value
yt  # tensor(9., grad_fn=<PowBackward0>): value calculated, a gradient function

yt.backward()  # tell PyTorch to calculate the gradients for us

xt.grad  # view the gradients ->  tensor(6.)
# The derivative of x**2 is 2*x, and we have x=3, so the gradients should be 2*3=6,
# which is what PyTorch calculated for us!

# repeat the preceding steps, but with a vector argument for our function:
# tensor([ 3., 4., 10.], requires_grad=True)
xt = tensor([3., 4., 10.], ).requires_grad_()

# add sum to our function so it can take a vector
# (i.e., a rank-1 tensor) and return a scalar (i.e., a rank-0 tensor):


def f2(x): return (x**2).sum()


yt = f2(xt)
# Our gradients are 2*xt, as we’d expect! => tensor(125., grad_fn=<SumBackward0>)
yt

yt.backward()
xt.grad  # tensor([ 6., 8., 20.])
