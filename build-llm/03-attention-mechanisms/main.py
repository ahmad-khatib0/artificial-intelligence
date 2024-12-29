import torch
import torch.nn as nn
# A simple self-attention mechanism without trainable weights

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
attn_scores_2 = torch.empty(inputs.shape[0])
print(attn_scores_2)
# tensor([7.1367e-24, 1.0286e-37, 4.1149e-37, 3.0045e-32, 2.7742e-32, 2.7742e-32])

for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)
# tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])

#  This normalization (obtain attention weights that sum up to 1)
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights: ", attn_weights_2_tmp)
print("Sum: ", attn_weights_2_tmp.sum())
# Attention weights:  tensor([0.1455, 0.2278, 0.2249, 0.1285, 0.1077, 0.1656])
# Sum:                tensor(1.0000)


# The following is a basic implementation of the softmax func for normalizing the attention scores:
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)


attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention weights: ", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())
# Attention weights:  tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
# Sum:                tensor(1.)


attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

# context vector z(2) is the weighted sum of all input vectors, obtained by
# multiplying each input vector by its corresponding attention weight:
query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
print(context_vec_2)  # tensor([0.4419, 0.6515, 0.5683])


# Computing attention weights for all input tokens
attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
        print(x_i, x_j)

print(attn_scores)
# tensor(
#     [
#         [0.9995, 0.9544, 0.9422, 0.4753, 0.4576, 0.6310],
#         [0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865],
#         [0.9422, 1.4754, 1.4570, 0.8296, 0.7154, 1.0605],
#         [0.4753, 0.8434, 0.8296, 0.4937, 0.3474, 0.6565],
#         [0.4576, 0.7070, 0.7154, 0.3474, 0.6654, 0.2935],
#         [0.6310, 1.0865, 1.0605, 0.6565, 0.2935, 0.9450],
#     ]
# )

# for loops are generally slow, and we can achieve the same results using matrix multiplication:
attn_scores = inputs @ inputs.T
print(attn_scores)  # same results

# we normalize each row so that the values in each row sum to 1:
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)
# tensor(
#     [
#         [0.2098, 0.2006, 0.1981, 0.1242, 0.1220, 0.1452],
#         [0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581],
#         [0.1390, 0.2369, 0.2326, 0.1242, 0.1108, 0.1565],
#         [0.1435, 0.2074, 0.2046, 0.1462, 0.1263, 0.1720],
#         [0.1526, 0.1958, 0.1975, 0.1367, 0.1879, 0.1295],
#         [0.1385, 0.2184, 0.2128, 0.1420, 0.0988, 0.1896],
#     ]
# )

row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum:", row_2_sum)
print("All row sums:", attn_weights.sum(dim=-1))
# Row 2 sum:    1.0
# All row sums: tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000])

# In the third and final step of figure 3.12, we use these attention
# weights to compute all context vectors via matrix multiplication:
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)
# tensor(
#     [
#         [0.4421, 0.5931, 0.5790],
#         [0.4419, 0.6515, 0.5683],
#         [0.4431, 0.6496, 0.5671],
#         [0.4304, 0.6298, 0.5510],
#         [0.4671, 0.5910, 0.5266],
#         [0.4177, 0.6503, 0.5645],
#     ]
# )


# We can double-check that the code is correct by comparing the
# second row with the context vector z(2) that we computed in section 3.3.1:
print("Previous 2nd context vector:", context_vec_2)
# Previous 2nd context vector: tensor([0.4419, 0.6515, 0.5683])


# Implementing self-attention with trainable weights (scaled dot-product attention)
# We will implement the self-attention mechanism step by step by introducing the three trainable
# weight matrices Wq, Wk, and Wv. These three matrices are used to project the embedded input
# tokens, x(i), into query, key, and value vectors, respectively, as illustrated in figure 3.14.


x_2 = inputs[1]
d_in = inputs.shape[1]  # The input embedding size, d=3
d_out = 2  # The output embedding size, d_out=2

# Next, we initialize the three weight matrices Wq, Wk, and Wv shown in figure 3.14:
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# Next, we compute the query, key, and value vectors:
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print(query_2)  # tensor([0.4306, 1.4551])

keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)
# keys.shape: torch.Size([6, 2])
# values.shape: torch.Size([6, 2])
# we successfully projected the six input tokens from a 3d onto a 2d embedding space

# First, let’s compute the attention score ω22:
keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)
# The result for the unnormalized attention score is =>  tensor(1.8524)

attn_scores_2 = query_2 @ keys.T  # All attention scores for given query
print(attn_scores_2)
# tensor([1.2705, 1.8524, 1.8111, 1.0795, 0.5577, 1.5440]) second el matches attn_score_22

# Now, we want to go from the attention scores to the attention weights, as illustrated in
# figure 3.16. We compute the attention weights by scaling the attention scores and
# using the softmax function. However, now we scale the attention scores by dividing
# them by the square root of the embedding dimension of the keys (taking the square
# root is mathematically the same as exponentiating by 0.5):
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)
# tensor([0.1500, 0.2264, 0.2199, 0.1311, 0.0906, 0.1820])

# we now compute the context vector as a weighted sum over the value vectors.
context_vec_2 = attn_weights_2 @ values
print(context_vec_2)  # tensor([0.3061, 0.8210])


class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        # The __init__ method initializes trainable weight matrices (W_query, W_key, and W_value) for
        # queries, keys, and values, each transforming the input dimension d_in to an output dimension d_out.
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T  # omega
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec


torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))
# tensor([[0.2996, 0.8053],
#         [0.3061, 0.8210],
#         [0.3058, 0.8203],
#         [0.2948, 0.7939],
#         [0.2927, 0.7891],
#         [0.2990, 0.8040]], grad_fn=<MmBackward0>)
# As a quick check, the second row ([0.3061, 0.8210]) matches the contents of context_vec_2


#
# Summary until now
# Self-attention involves the trainable weight matrices W q, Wk, and Wv. These matrices transform input
# data into queries, keys, and values, respectively, which are crucial components of the attention
# mechanism. As the model is exposed to more data during training, it adjusts these trainable weights,
# as we will see in upcoming chapters. We can improve the SelfAttention_v1 implementation further by
# utilizing PyTorch’s nn.Linear layers, which effectively perform matrix multiplication when the bias
# units are disabled. Additionally, a significant advantage of using nn.Linear instead of manually
# implementing nn.Parameter(torch.rand(...)) is that nn.Linear has an optimized weight initialization
# scheme, contributing to more stable and effective model training.


# A self-attention class using PyTorch’s Linear layers
# instead of manually implementing nn.Parameter(torch.rand(...)). nn.Linear has an optimized weight
# initialization scheme, contributing to more stable and effective model training.
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec


torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print(sa_v2(inputs))


queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
print(attn_weights)

# create a mask where the values above the diagonal are zero:
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)
# tensor(
#     [
#         [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
#         [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
#         [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
#         [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
#         [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
#     ]
# )


# Now, we can multiply this mask with the attention weights to zero-out the values above the diagonal:
masked_simple = attn_weights * mask_simple
print(masked_simple)
# tensor(
#     [
#         [0.1921, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.2041, 0.1659, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.2036, 0.1659, 0.1662, 0.0000, 0.0000, 0.0000],
#         [0.1869, 0.1667, 0.1668, 0.1571, 0.0000, 0.0000],
#         [0.1830, 0.1669, 0.1670, 0.1588, 0.1658, 0.0000],
#         [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529],
#     ],
#     grad_fn=<MulBackward0>
# )


# The third step is to renormalize the attention weights to sum up to 1 again in each row.
# We can achieve this by dividing each element in each row by the sum in each row:
row_sums = masked_simple.sum(dim=-1, keepdim=True)
print(row_sums)
# tensor([[0.1921], [0.3700], [0.5357], [0.6775], [0.8415], [1.0000]], grad_fn=<SumBackward1>)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)
# tensor(
#     [
#         [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.5517, 0.4483, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.3800, 0.3097, 0.3103, 0.0000, 0.0000, 0.0000],
#         [0.2758, 0.2460, 0.2462, 0.2319, 0.0000, 0.0000],
#         [0.2175, 0.1983, 0.1984, 0.1888, 0.1971, 0.0000],
#         [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529],
#     ],
#     grad_fn=<DivBackward0>
# )

# We can implement this more efficient masking “trick” by creating a mask with 1s
# above the diagonal and then replacing these 1s with negative infinity (-inf) values:
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)
# tensor(
#     [
#         [0.2899, -inf, -inf, -inf, -inf, -inf],
#         [0.4656, 0.1723, -inf, -inf, -inf, -inf],
#         [0.4594, 0.1703, 0.1731, -inf, -inf, -inf],
#         [0.2642, 0.1024, 0.1036, 0.0186, -inf, -inf],
#         [0.2183, 0.0874, 0.0882, 0.0177, 0.0786, -inf],
#         [0.3408, 0.1270, 0.1290, 0.0198, 0.1290, 0.0078],
#     ],
#     # grad_fn=<MaskedFillBackward0>
# )

attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=1)
print(attn_weights)
# based on the output, the values in each row sum to 1, and no further normalization is necessary:
# tensor(
#     [
#         [1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.5517, 0.4483, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.3800, 0.3097, 0.3103, 0.0000, 0.0000, 0.0000],
#         [0.2758, 0.2460, 0.2462, 0.2319, 0.0000, 0.0000],
#         [0.2175, 0.1983, 0.1984, 0.1888, 0.1971, 0.0000],
#         [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529],
#     ],
#     grad_fn=<SoftmaxBackward0>
# )


# Masking additional attention weights with dropout:
# a dropout rate of 50%, which means masking out half of the attention weights.
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6, 6)
print(dropout(example))
# tensor(
#     [
#         [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
#         [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0, 2.0, 0.0, 2.0, 0.0],
#         [2.0, 2.0, 0.0, 0.0, 0.0, 2.0],
#         [2.0, 0.0, 0.0, 0.0, 0.0, 2.0],
#         [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
#     ]
# )
# When applying dropout to an attention weight matrix with a rate of 50%, half of the elements in the
# matrix are randomly set to zero. To compensate for the reduction in active elements, the values of the
# remaining elements in the matrix are scaled up by a factor of 1/0.5 = 2. This scaling is crucial to
# maintain the overall balance of the attention weights, ensuring that the average influence of the
# attention mechanism remains consistent during both the training and inference phases.

torch.manual_seed(123)
print(dropout(attn_weights))
# tensor(
#     [
#         [2.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.0000, 0.8966, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.0000, 0.0000, 0.6206, 0.0000, 0.0000, 0.0000],
#         [0.5517, 0.4921, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.4350, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.0000, 0.3327, 0.0000, 0.0000, 0.0000, 0.0000],
#     ],
#     grad_fn=<MulBackward0>
# )


# Implementing a compact causal attention class:
# We duplicate the inputs 2 times to simulate such batch inputs
batch = torch.stack((inputs, inputs), dim=0)
print(batch.shape)  # torch.Size([2, 6, 3])
# This results in a three-dimensional tensor consisting of two input texts with
# six tokens each, where each token is a three-dimensional embedding vector


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)  # New
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We transpose dimensions 1 and 2, keeping the batch dimension at the first position (0).
        attn_scores = queries @ keys.transpose(1, 2)  # Changed transpose
        # In PyTorch, operations with a trailing underscore are
        # performed in-place, avoiding unnecessary memory copies.
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)  # New

        context_vec = attn_weights @ values
        return context_vec


torch.manual_seed(123)
context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)
# context_vecs.shape: torch.Size([2, 6, 2])


# 3.6.1 Extending single-head attention to multi-head attention
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
                for _ in range(num_heads)
            ]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


# E,g, if we use this MultiHeadAttentionWrapper class with two attention heads (via num_heads=2) and
# CausalAttention output dimension d_out=2, we get a fourdimensional context vector (d_out*num_heads=4)

torch.manual_seed(123)
print(batch.shape)
context_length = batch.shape[1]  # This is the number of tokens
d_in, d_out = 3, 2


mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)  # torch.Size([2, 6, 4])
# The first dimension of the resulting context_vecs tensor is 2 since we have two input texts
# (the input texts are duplicated, which is why the context vectors are exactly the
# same for those). The second dimension refers to the 6 tokens in each input. The third
# dimension refers to the four-dimensional embedding of each token.


# Implementing multi-head attention with weight splits
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        # Reduces the projection dim to match the desired output dim
        self.head_dim = d_out // num_heads
        print("head_dim", self.head_dim)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # Uses a Linear layer to combine head outputs
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask", torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        # Tensor shape: (b, num_tokens, d_out)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        print("values before views", values)

        # We implicitly split the matrix by adding a num_heads dimension. Then we unroll the
        # last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim).
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        print("values after views", values)

        # Transposes from shape (b, num_tokens, num_heads, head_dim) to (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        print("values after transpose", values)

        # Computes dot product for each head
        attn_scores = queries @ keys.transpose(2, 3)
        # Masks truncated to the number of tokens
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # Uses the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # Tensor shape: (b, num_tokens, n_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)
        # Combines heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # Adds an optional linear projection
        context_vec = self.out_proj(context_vec)
        return context_vec


torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)

# tensor(
#     [
#         [
#             [0.3190, 0.4858],
#             [0.2943, 0.3897],
#             [0.2856, 0.3593],
#             [0.2693, 0.3873],
#             [0.2639, 0.3928],
#             [0.2575, 0.4028],
#         ],
#         [
#             [0.3190, 0.4858],
#             [0.2943, 0.3897],
#             [0.2856, 0.3593],
#             [0.2693, 0.3873],
#             [0.2639, 0.3928],
#             [0.2575, 0.4028],
#         ],
#     ],
#     grad_fn=<ViewBackward0>
# )
# context_vecs.shape: torch.Size([2, 6, 2])
