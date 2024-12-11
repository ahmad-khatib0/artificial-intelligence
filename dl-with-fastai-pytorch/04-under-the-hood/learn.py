from fastai.vision.all import *
from string import ascii_lowercase

# To create an array or tensor, pass a list (or list of lists, or list of
# lists of lists, etc.) to array or tensor:

data = [[1, 2, 3], [4, 5, 6]]
arr = array(data)
tns = tensor(arr)

arr
# numpy: array([[1, 2, 3], [4, 5, 6]])

tns
# pytorch: tensor([[1, 2, 3], [4, 5, 6]])

# All the operations that follow are shown on tensors, but the
# syntax and results for NumPy arrays are identical.
tns[1]  # select row: tensor([4, 5, 6])

tns[:, 1]  # a column, (: to indicate all of the first axis)

# You can combine these with Python slice syntax ([start:end],
# with end being excluded) to select part of a row or column:
tns[1, 1:3]  # tensor([5, 6])

# And you can use the standard operators, such as +, -, *, and /:
tns + 1  # tensor([[2, 3, 4], [5, 6, 7]])

# Tensors have a type:
tns.type()  # 'torch.LongTensor'

# And will automatically change that type as needed; for example, from int to float:
tns*1.5  # tensor([[1.5000, 3.0000, 4.5000], [6.0000, 7.5000, 9.0000]])

tensor([1, 2, 3]) + tensor([1, 1, 1])  # tensor([2, 3, 4])


# extremely simple Dataset:
ds = L(enumerate(string, ascii_lowercase))
ds
# (#26) [(0, 'a'),(1, 'b'),(2, 'c'),(3, 'd'),(4, 'e'),(5, 'f'),(6, 'g'),(7, > 'h'),(8, 'i'),(9, 'j')...]
