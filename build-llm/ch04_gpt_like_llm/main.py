import torch
import tiktoken
import torch.nn as nn
import matplotlib.pyplot as plt
from prev_multihead import MultiHeadAttention


GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False,  # Query-Key-Value bias
}


# A placeholder GPT model architecture class
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        # Uses a placeholder for TransformerBlock
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        # Uses a placeholder for LayerNorm
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        print("in_idx is: \n", in_idx)
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


# A simple placeholder class that will be replaced by a real TransformerBlock later
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

    def forward(self, x):
        return x  # does nothing, just returns its input.


# A simple placeholder class that will be replaced by a real LayerNorm later
class DummyLayerNorm(nn.Module):
    # The parameters here are just to mimic the LayerNorm interface.
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x


tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)
# tensor([[6109, 3626, 6100, 345], [6109, 1110, 6622, 257]])
# The first row corresponds to the first text, and the second row corresponds to the second text.

torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)  # a new 124-million-parameter DummyGPTModel
logits = model(batch)
print("Output shape", logits.shape)  # Output shape torch.Size([2, 4, 50257])
print(logits)
# tensor(
#     [
#         [
#             [-0.9289, 0.2748, -0.7557, ..., -1.6070, 0.2702, -0.5888],
#             [-0.4476, 0.1726, 0.5354, ..., -0.3932, 1.5285, 0.8557],
#             [0.5680, 1.6053, -0.2155, ..., 1.1624, 0.1380, 0.7425],
#             [0.0448, 2.4787, -0.8843, ..., 1.3219, -0.0864, -0.5856],
#         ],
#         [
#             [-1.5474, -0.0542, -1.0571, ..., -1.8061, -0.4494, -0.6747],
#             [-0.8422, 0.8243, -0.1098, ..., -0.1434, 0.2079, 1.2046],
#             [0.1355, 1.1858, -0.1453, ..., 0.0869, -0.1590, 0.1552],
#             [0.1666, -0.8138, 0.2307, ..., 2.5035, -0.3055, -0.3083],
#         ],
#     ],
#     # grad_fn=<UnsafeViewBackward0>
# )
# The output tensor has two rows corresponding to the two text samples. Each text sample consists
# of four tokens; each token is a 50,257-dimensional vector, which matches the size of the tokenizer’s
# vocabulary. The embedding has 50,257 dimensions because each of these dimensions refers to a unique
# token in the vocabulary. When we implement the postprocessing code, we will convert these
# 50,257-dimensional vectors back into token IDs, which we can then decode into words.


# five inputs and six outputs that we apply to two input examples
torch.manual_seed(123)
# Creates two training examples with five dimensions (features) each
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print(out)
# tensor(
#     [
#         [0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000],
#         [0.2133, 0.2394, 0.0000, 0.5198, 0.3297, 0.0000],
#     ],
#     # grad_fn=<ReluBackward0>
# )
# the first row lists the layer outputs for the first input and
# the second row lists the layer outputs for the second row

# Before we apply layer normalization to these outputs, let’s examine the mean and variance:
mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)

print("Mean:\n", mean)  #    tensor([[0.1324], [0.2170]], grad_fn=<MeanBackward1>)
# The first row in the mean tensor here contains the mean value for the first
# input row, and the second output row contains the mean for the second input row.

print("Variance:\n", var)  # tensor([[0.0231], [0.0398]], grad_fn=<VarBackward0>)

# The first row in the mean tensor here contains the mean value for the first input row, and the
# second output row contains the mean for the second input row. Using keepdim=True in operations
# like mean or variance calculation ensures that the output tensor retains the same number of
# dimensions as the input tensor, even though the operation reduces the tensor along the dimension
# specified via dim. For instance, without keepdim=True, the returned mean tensor would be a
# two-dimensional vector [0.1324, 0.2170] instead of a 2 × 1–dimensional matrix [[0.1324], [0.2170]].
# The dim parameter specifies the dimension along which the calculation of the statistic
# (here, mean or variance) should be performed in a tensor. As figure 4.6 explains, for a
# two-dimensional tensor (like a matrix), using dim=-1 for operations such as mean or variance
# calculation is the same as using dim=1. This is because -1 refers to the tensor’s last dimension,
# which corresponds to the columns in a two-dimensional tensor. Later, when adding layer normalization
# to the GPT model, which produces three-dimensional tensors with the shape:
# [batch_size, num_tokens, embedding_size], we can still use dim=-1 for normalization across the
# last dimension, avoiding a change from dim=1 to dim=2.


# Next, let’s apply layer normalization to the layer outputs we obtained earlier.
# The operation consists of subtracting the mean and dividing by the square root
# of the variance (also known as the standard deviation):
out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
print("Normalized layer outputs:\n", out_norm)
# tensor(
#     [
#         [0.6159, 1.4126, -0.8719, 0.5872, -0.8719, -0.8719],
#         [-0.0189, 0.1121, -1.0876, 1.5173, 0.5647, -1.0876],
#     ],
#     grad_fn=<DivBackward0>
# )
print("Mean:\n", mean)
# tensor([[9.9341e-09], [1.9868e-08]], grad_fn=<MeanBackward1>)
print("Variance:\n", var)
# tensor([[1.0000], [1.0000]], grad_fn=<VarBackward0>)
# 0 mean and a variance of 1

# To improve readability, we can also turn off the scientific notation when printing tensor
torch.set_printoptions(sci_mode=False)
print("Mean:\n", mean)
print("Variance:\n", var)
# tensor([[0.0000], [0.0000]], grad_fn=<MeanBackward1>)
# tensor([[1.0000], [1.0000]], grad_fn=<VarBackward0>)


# This specific implementation of layer normalization operates on the last dimension of the input
# tensor x, which represents the embedding dimension (emb_dim). The variable eps is a small constant
# (epsilon) added to the variance to prevent division by zero during normalization. The scale and
# shift are two trainable parameters (of the same dimension as the input) that the LLM automatically
# adjusts during training if it is determined that doing so would improve the model’s performance on
# its training task. This allows the model to learn appropriate scaling and shifting that best suit
# the data it is processing.
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=True)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim=-1, keepdim=True)
out = out_ln.var(dim=-1, keepdim=True, unbiased=False)
print("Mean:\n", mean)
print("Variance:\n", var)

# The results show that the layer normalization code works as expected and normalizes
# the values of each of the two inputs such that they have a mean of 0 and a variance of 1:
# tensor([[-0.0000], [0.0000]], grad_fn=<MeanBackward1>)
# tensor([[1.0000],  [1.0000]], grad_fn=<VarBackward0>)


# An implementation of the GELU activation function
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )


# Next, to get an idea of what this GELU function looks like and how it
# compares to the ReLU function, let’s plot these functions side by side:
gelu, relu = GELU(), nn.ReLU()

# Creates 100 sample data points in the range -3 to 3
x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)

plt.figure(figsize=(8, 3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)

plt.tight_layout()
# plt.show()


class FeedForward(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            # Input tensor with shape (2, 3, 768)
            # The three values represent the batch size (2), number of tokens (3), and embedding size (768).
            #
            # The first linear layer increases the embedding dimension by a factor of 4.
            # Input: (2, 3, 768) Output: (2, 3, 3072)
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            # Input: (2, 3, 3072) Output: (2, 3, 3072
            GELU(),
            # The second linear layer decreases the embedding dimension by a factor of 4.
            # Input: (2, 3, 3072) Output: (2, 3, 768)
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


# Creates sample input with batch dimension 2
ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
out = ffn(x)
print(out.shape)  # torch.Size([2, 3, 768])


class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut) -> None:
        super().__init__()
        self.use_shortcut = use_shortcut
        # implement 5 layers
        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            # Compute the output of the current layer
            layer_output = layer(x)
            # check if shortcut can be applied
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output

        return x


# Let’s use this code to initialize a neural network without shortcut connections.
# Each layer will be initialized such that it accepts an example with three input values
# and returns three output values. The last layer returns a single output value:
layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1.0, 0.0, -1.0]])
torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)


# Next, we implement a function that computes the gradients in the model’s backward pass:
def print_gradients(model, x):
    """
    This code specifies a loss function that computes how close the model output and a user-specified
    target (here, for simplicity, the value 0) are. Then, when calling loss.backward(), PyTorch computes
    the loss gradient for each layer in the model. We can iterate through the weight parameters via
    model.named_parameters(). Suppose we have a 3 × 3 weight parameter matrix for a given layer.
    In that case, this layer will have 3 × 3 gradient values, and we print the mean absolute gradient
    of these 3 × 3 gradient values to obtain a single gradient value per layer to compare the
    gradients between layers more easily.
    """
    output = model(x)  # forward pass
    target = torch.tensor([[0.0]])

    loss = nn.MSELoss()
    # Calculates loss based on how close the target and output are
    loss = loss(output, target)

    # Backward pass to calculate the gradients
    loss.backward()

    for name, param in model.named_parameters():
        if "weight" in name:
            #  Print the mean absolute gradient of the weights
            print(f"{name} has gradients mean of {param.grad.abs().mean().item()}")


print_gradients(model_without_shortcut, sample_input)
# layers.0.0.weight has gradients mean of 0.00020173584925942123
# layers.1.0.weight has gradients mean of 0.00012011159560643137
# layers.2.0.weight has gradients mean of 0.0007152040489017963
# layers.3.0.weight has gradients mean of 0.0013988736318424344
# layers.4.0.weight has gradients mean of 0.005049645435065031
#
# The output of the print_gradients function shows, the gradients become smaller
# as we progress from the last layer ( layers.4 ) to the first layer (layers.0),
# which is a phenomenon called the vanishing gradient problem.

torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
print_gradients(model_with_shortcut, sample_input)
# layers.0.0.weight has gradient mean of 0.22169792652130127
# layers.1.0.weight has gradient mean of 0.20694105327129364
# layers.2.0.weight has gradient mean of 0.32896995544433594
# layers.3.0.weight has gradient mean of 0.2665732502937317
# layers.4.0.weight has gradient mean of 1.3258541822433472
#
# The last layer (layers.4) still has a larger gradient than the other layers. However,
# the gradient value stabilizes as we progress toward the first layer (layers.0) and
# doesn’t shrink to a vanishingly small value.


### Connecting attention and linear layers in a transformer block
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    # Layer normalization (LayerNorm ) is applied before each of these two components,
    # and dropout is applied after them to regularize the model and prevent overfitting
    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


torch.manual_seed(123)
# Creates sample input of shape [batch_size, num_tokens, emb_dim]
x = torch.rand(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

print("Input shape:", x.shape)  #       Input shape: torch.Size([2, 4, 768])
print("Output shape:", output.shape)  # Output shape: torch.Size([2, 4, 768])
# As we can see, the transformer block maintains the input dimensions in its output,
# indicating that the transformer architecture processes sequences of data without
# altering their shape throughout the network


# The GPT model architecture implementation (Figure 4.15)
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        # LayerNorm for standardizing the outputs from the transformer blocks to stabilize the learning process.
        self.final_norm = LayerNorm(cfg["emb_dim"])
        # projects the transformer’s output into the vocabulary space of
        # the tokenizer to generate logits for each token in the vocabulary.
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        # The device setting will allow us to train the model on a CPU or GPU,
        # depending on which device the input data sits on.
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        # computes the logits, representing the next token’s unnormalized probabilities.
        logits = self.out_head(x)
        return logits


torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

out = model(batch)
print("Input batch:\n", batch)
# tensor([[6109, 3626, 6100, 345], [6109, 1110, 6622, 257]])  Token IDs of text 1, Token IDs of text 2
print("\nOutput shape:", out.shape)  # Output shape: torch.Size([2, 4, 50257])
print(out)
# tensor(
#     [
#         [
#             [0.3613, 0.4222, -0.0711, ..., 0.3483, 0.4661, -0.2838],
#             [-0.1792, -0.5660, -0.9485, ..., 0.0477, 0.5181, -0.3168],
#             [0.7120, 0.0332, 0.1085, ..., 0.1018, -0.4327, -0.2553],
#             [-1.0076, 0.3418, -0.1190, ..., 0.7195, 0.4023, 0.0532],
#         ],
#         [
#             [-0.2564, 0.0900, 0.0335, ..., 0.2659, 0.4454, -0.6806],
#             [0.1230, 0.3653, -0.2074, ..., 0.7705, 0.2710, 0.2246],
#             [1.0558, 1.0318, -0.2800, ..., 0.6936, 0.3205, -0.3178],
#             [-0.1565, 0.3926, 0.3288, ..., 1.2630, -0.1858, 0.0388],
#         ],
#     ],
#     grad_fn=<UnsafeViewBackward0>
# )

#  numel() method, short for “number of elements,”
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")
# Total number of parameters: 163,009,536 # why it's not a 124-million-parameter ?

# The reason is a concept called weight tying, which was used in the original GPT-2 architecture. It means that the
# original GPT-2 architecture reuses the weights from the token embedding layer in its output layer. To understand
# better, let’s take a look at the shapes of the token embedding layer and linear output layer
print("Token embedding layer shape: ", model.tok_emb.weight.shape)
print("Output layer shape: ", model.out_head.weight.shape)
# Token embedding layer shape: torch.Size([50257, 768])
# Output layer shape: torch.Size([50257, 768])

# The token embedding and output layers are very large due to the number of rows for the 50,257
# in the tokenizer’s vocabulary. Let’s remove the output layer parameter count from the total
# GPT-2 model count according to the weight tying:
total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
print(
    f"Number of trainable parameters  considering weight tying: {total_params_gpt2:,}"
)
# Number of trainable parameters considering weight tying: 124,412,160

# Lastly, let’s compute the memory requirements of the 163 million parameters in our GPTModel object:

# Calculates the total size in bytes (assuming float32, 4 bytes per parameter)
total_size_bytes = total_params * 4
total_size_mb = total_size_bytes / (1024 * 1024)
print(f"Total size of the model: {total_size_mb:.2f} MB")  # 621.83 MB


# Figure 4.17
def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    idx is a (batch, n_tokens) array of indices in the current context.
    """
    for _ in range(max_new_tokens):
        # Crops current context if it exceeds the supported context size, e.g., if LLM supports only 5
        # tokens, and the context size is 10, then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        # Focuses only on the last time step, so that (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]
        # probas has shape (batch, vocab_size).
        probas = torch.softmax(logits, dim=-1)
        # idx_next has shape (batch, 1).
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        # Appends sampled index to the running sequence, where idx has shape (batch, n_tokens+1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded: ", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # Adds batch dimension
# print("encoded_tensor.shape:", encoded_tensor.shape)
# encoded: [15496, 11, 314, 716]
# encoded_tensor.shape: torch.Size([1, 4])

# Next, we put the model into .eval() mode. This disables random components like dropout, which are
# only used during training, and use the generate_text_simple function on the encoded input tensor:
model.eval()  # Disables dropout since we are not training the model
out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"],
)

print("Output:", out)
print("Output length:", len(out[0]))
# Output: tensor([[15496,    11,   314,   716, 27018, 24086, 47843, 30961, 42348,  7267]])
# Output length: 10


decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)
# Note that the model is untrained; hence the random output texts above
# We will train the model in the next chapter
