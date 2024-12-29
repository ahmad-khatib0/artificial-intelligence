import torch

inputs = torch.tensor(
    [
        [0.43, 0.15, 0.89],  # Your (x^1)
        [0.55, 0.87, 0.66],  # journey (x^2)
        [0.57, 0.85, 0.64],  # starts (x^3)
        [0.22, 0.58, 0.33],  # with (x^4)
        [0.77, 0.25, 0.10],  # one (x^5)
        [0.05, 0.80, 0.55],  # step (x^6)
    ]
)

query = inputs[1]  #  The second input token serves as the query.

res = 0.0
for idx, element in enumerate(inputs[0]):
    res += inputs[0][idx] * query[idx]
    print(element, query[idx])

print(res)
print(torch.dot(inputs[0], query))
# The output confirms that the sum of the element-wise multiplication gives the same
# results as the dot product:
# tensor(0.9544)
# tensor(0.9544)


# view() reshapes the tensor without copying memory, similar to numpy's reshape().
a = torch.arange(1, 17)
a = a.view(4, 4)
print(a)


# The T
a = torch.arange(1, 17)
a = a.view(4, 4)
print("without t parameter \n", a, "\n the t parameter \n", a.T)
# tensor([
#     [1, 2, 3, 4],
#     [5, 6, 7, 8],
#     [9, 10, 11, 12],
#     [13, 14, 15, 16]
# ])
# tensor([
#     [1, 5, 9, 13], # this 1st row is the 1st column in the origional matrix
#     [2, 6, 10, 14], # this 2nd row is the 2nd column in the origional matrix
#     [3, 7, 11, 15], # ....
#     [4, 8, 12, 16]
# ])


####
# The shape of this tensor is (b, num_heads, num_tokens, head_dim) = (1, 2, 3, 4).
a = torch.tensor(
    [
        [
            [
                [0.2745, 0.6584, 0.2775, 0.8573],
                [0.8993, 0.0390, 0.9268, 0.7388],
                [0.7179, 0.7058, 0.9156, 0.4340],
            ],
            [
                [0.0772, 0.3565, 0.1479, 0.5331],
                [0.4066, 0.2318, 0.4545, 0.9737],
                [0.4606, 0.5159, 0.4220, 0.5786],
            ],
        ]
    ]
)

print(a @ a.transpose(2, 3))
# tensor(
#     [
#         [
#             [
#                 [1.3208, 1.1631, 1.2879],
#                 [1.1631, 2.2150, 1.8424],
#                 [1.2879, 1.8424, 2.0402],
#             ],
#             [
#                 [0.4391, 0.7003, 0.5903],
#                 [0.7003, 1.3737, 1.0620],
#                 [0.5903, 1.0620, 0.9912],
#             ],
#         ]
#     ]
# )
# In this case, the matrix multiplication implementation in PyTorch handles the four-dimensional
# input tensor so that the matrix multiplication is carried out between the two
# last dimensions (num_tokens, head_dim) and then repeated for the individual heads.

# For instance, the preceding becomes a more compact way to compute the matrix
# multiplication for each head separately:
first_head = a[0, 0, :, :]
first_res = first_head @ first_head.T
print("First head:\n", first_res)

second_head = a[0, 1, :, :]
second_res = second_head @ second_head.T
print("\nSecond head:\n", second_res)
# The results are exactly the same results as those we obtained when
# using the batched matrix multiplication print(a @ a.transpose(2, 3)):
# First head:
# tensor([
#     [1.3208, 1.1631, 1.2879],
#     [1.1631, 2.2150, 1.8424],
#     [1.2879, 1.8424, 2.0402]
# ])
# # Second head:
# tensor([
#     [0.4391, 0.7003, 0.5903],
#     [0.7003, 1.3737, 1.0620],
#     [0.5903, 1.0620, 0.9912]
# ])
