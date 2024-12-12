from fastbook import F, nn, plot_function, tensor, torch


acts = torch.randn((6, 2)) * 2

sm_acts = torch.softmax(acts, dim=1)
sm_acts
# tensor([[0.6025, 0.3975], ..., [0.3661,0.6339]]) # adds up to 1

# example
targ = tensor([0, 1, 0, 1, 1, 0])
sm_acts
idx = range(6)
sm_acts[idx,  targ]
# tensor([0.6025, 0.4979, 0.1332, 0.0034, 0.4041, 0.3661]) (these are the losses)


F.nll_loss(sm_acts, targ, reduction='none')
# tensor([-0.6025, -0.4979, -0.1332, -0.0034, -0.4041, -0.3661])


plot_function(torch.log, min=0, max=4)

loss_func = nn.CrossEntropyLoss()  # class way
loss_func(acts, targ)
# tensor(1.8045)

F.cross_entropy(acts, targ)  # functionaly way
tensor(1.8045)

# By default, PyTorch loss functions take the mean of the loss of all items. to disable that:
nn.CrossEntropyLoss(reduction='none')(acts, targ)
# tensor([0.5067, 0.6973, 2.0160, 5.6958, 0.9062, 1.0048])
