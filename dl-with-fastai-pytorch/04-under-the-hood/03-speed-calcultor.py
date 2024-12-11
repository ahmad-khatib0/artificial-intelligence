from fastai.vision.all import plt, to_np, torch


# Imagine you were measuring the speed of a roller coaster as it went over the top of a hump. It would start
# fast, and then get slower as it went up the hill; it would be slowest at the top, and it would then speed up
# again as it went downhill. You want to build a model of how the speed changes over time. If you were
# measuring the speed manually every second for 20 seconds, it might look something like this:


time = torch.arange(0, 20).float()
time  # tensor([ 0., 1., ... , 18., 19.])

speed = torch.randn(20)*3 + 0.75*(time-9.5)**2 + 1
plt.scatter(time, speed)


# added a bit of random noise, since measuring things manually isn’t precise This means it’s not that
# easy to answer the question: what was the roller coaster’s speed? Using SGD, we can try to find a
# function that matches our observations. We can’t consider every possible function, so let’s use a
# guess that it will be quadratic; i.e., a function of the form a*(time**2)+ (b*time)+c.


def f(t, params):
    # We want to distinguish clearly between the function’s input (the time when we are measuring
    # the coaster’s speed (t)) and its parameters (the values that define which quadratic we’re trying(params))
    a, b, c = params
    return a*(t**2) + (b*t) + c


def mse(preds, targets): return ((preds - targets) ** 2).mean()


# STEP 1: INITIALIZE THE PARAMETERS (pytorch track their gradients using requires_grad_):
params = torch.randn(3).requires_grad_()

# STEP 2: CALCULATE THE PREDICTIONS
preds = f(time, params)


def show_preds(preds, ax=None):
    # function to see how close our predictions are to our targets
    if ax is None:
        ax = plt.subplots()[1]
    ax.scatter(time, speed)
    ax.scatter(time, to_np(preds), color='red')
    ax.set_ylim(-300, 100)


show_preds(preds)

# STEP 3: CALCULATE THE LOSS
loss = mse(preds, speed)
loss  # tensor(25823.8086, grad_fn=<MeanBackward0>)


# STEP 4: CALCULATE THE GRADIENTS
loss.backward()
params.grad  # tensor([-53195.8594, -3419.7146, -253.8908])

params.grad * 1e-5  # tensor([-0.5320, -0.0342, -0.0025])


# We can use these gradients to improve our parameters. We’ll need to pick a learning rate (we’ll discuss
# how to do that in practice in the next chapter; for now, we’ll just use 1e-5 or 0.00001):
params  # tensor([-0.7658, -0.7506, 1.3525], requires_grad=True)

# STEP 5: STEP THE WEIGHTS
# Now we need to update the parameters based on the gradients we just calculated:
lr = 1e-5
params.data -= lr * params.grad.data
params.grad = None

# Let’s see if the loss has improved:
preds = f(time, params)
mse(preds, speed)
# tensor(5435.5366, grad_fn=<MeanBackward0>)

show_preds(preds)


def apply_step(params, prn=True):
    # We need to repeat this a few times, so we’ll create a function to apply one step:
    preds = f(time, params)
    loss = mse(preds, speed)
    loss.backward()
    params.data -= lr * params.grad.data
    params.grad = None
    if prn:
        print(loss.item())
    return preds


# STEP 6: REPEAT THE PROCESS
# Now we iterate. By looping and performing many improvements, we hope to reach a good result:
for i in range(10):
    apply_step(i)
# 5435.53662109375
# 1577.4495849609375
# 847.3780517578125
# 709.22265625
# 683.0757446289062
# 678.12451171875
# 677.1839599609375
# 677.0025024414062
# 676.96435546875
# 676.9537353515625

 # on the way to finding the best possible quadratic function. We can see this process visually:
_, axs = plt.subplots(1, 4, figsize=(12, 3))
for ax in axs:
    show_preds(apply_step(params, False), ax)
plt.tight_layout()

# STEP 7: STOP
# We just decided to stop after 10 epochs arbitrarily. In practice, we would watch the training
# and validation losses and our metrics to decide when to stop, as we’ve discussed.
