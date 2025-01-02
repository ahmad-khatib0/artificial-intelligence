import torch


# creating the input tensor
input = torch.randn(3, 1, 2, 1, 4)

print(input)
# tensor(
#     [
#         [[[[1.1727, -0.0411, -1.0675, 0.6670]], [[-0.6076, 0.1565, 1.1264, -0.4478]]]],
#         [[[[-0.5239, 1.4084, -0.1291, -0.9222]], [[-0.3125, -0.1399, 0.2045, 1.1794]]]],
#         [[[[0.0379, 0.5081, -0.8218, 0.6166]], [[-0.2586, 0.1359, -1.0692, 1.1898]]]],
#     ]
# )

print("Input tensor Size:\n", input.size())
# Input tensor Size: torch.Size([3, 1, 2, 1, 4])

output = torch.squeeze(input)
print("Size after squeeze:\n", output.size())
# Size after squeeze: torch.Size([3, 2, 4])
print(output)
# tensor(
#     [
#         [[-0.1415, -2.1716, 0.8031, 1.2232], [0.4886, -0.5850, 0.5672, 0.6095]],
#         [[1.1777, 0.5294, -1.0300, 0.9848], [0.2569, 0.4981, 1.7220, 0.5566]],
#         [[-0.3535, -0.0608, -0.2284, -0.0695], [1.1294, -0.1907, -0.0759, -0.5524]],
#     ]
# )
# Notice that both dimensions of size 1 are removed in the squeezed tensor.

# Example 2:
# In this example, We squeeze the tensor into different dimensions.
input = torch.randn(3, 1, 2, 1, 4)
print("Dimension of input tensor:", input.dim())  # 5
print("Input tensor Size:\n", input.size())  #  torch.Size([3, 1, 2, 1, 4])

# squeeze the tensor in dimension 0
output = torch.squeeze(input, dim=0)
print("Size after squeeze with dim=0:\n", output.size())
# Size after squeeze with dim=0: torch.Size([3, 1, 2, 1, 4])

# squeeze the tensor in dimension 1
output = torch.squeeze(input, dim=1)
print("Size after squeeze with dim=1:\n", output.size())
# Size after squeeze with dim=1: torch.Size([3, 2, 1, 4])

# squeeze the tensor in dimension 2
output = torch.squeeze(input, dim=2)
print("Size after squeeze with dim=2:\n", output.size())
# Size after squeeze with dim=2: torch.Size([3, 1, 2, 1, 4])

# squeeze the tensor in dimension 3
output = torch.squeeze(input, dim=3)
print("Size after squeeze with dim=3:\n", output.size())
# Size after squeeze with dim=3: torch.Size([3, 1, 2, 4])

# squeeze the tensor in dimension 4
output = torch.squeeze(input, dim=4)
print("Size after squeeze with dim=4:\n", output.size())
# Size after squeeze with dim=4: torch.Size([3, 1, 2, 1, 4])

# output = torch.squeeze(input,dim=5) # Error

# Notice that when we squeeze the tensor in dimension 0, there is no change in the shape of the
# output tensor. When we squeeze in dimension 1 or in dimension 3 (both are of size 1), only this
# dimension is removed in the output tensor. When we squeeze in dimension 2 or in dimension 4,
# there is no change in the shape of the output tensor.


# Unsqueeze a Tensor:
# When we unsqueeze a tensor, a new dimension of size 1 is inserted at the specified position.
# Always an unsqueeze operation increases the dimension of the output tensor.
# For example, if the input tensor is of shape:  (m×n) and we want to insert a new dimension
# at position 1 then the output tensor after unsqueeze will be of shape: (m×1×n). The
# following is the syntax of the torch.unsqueeze() method


# Example 3:
# In the example below we unsqueeze a 1-D tensor to a 2D tensor.
# define the input tensor
input = torch.arange(8, dtype=torch.float)
print("Input tensor:\n", input)
# Input tensor: tensor([0., 1., 2., 3., 4., 5., 6., 7.])
print("Size of input Tensor before unsqueeze:\n", input.size())
# Size of input Tensor before unsqueeze: torch.Size([8])

output = torch.unsqueeze(input, dim=0)
print("Tensor after unsqueeze with dim=0:\n", output)
# tensor([[0., 1., 2., 3., 4., 5., 6., 7.]])
print("Size after unsqueeze with dim=0:\n", output.size())
# Size after unsqueeze with dim=0: torch.Size([1, 8])

output = torch.unsqueeze(input, dim=1)
print("Tensor after unsqueeze with dim=1:\n", output)
# tensor([[0.], [1.], [2.], [3.], [4.], [5.], [6.], [7.]])
print("Size after unsqueeze with dim=1:\n", output.size())
# Size after unsqueeze with dim=1: torch.Size([8, 1])
