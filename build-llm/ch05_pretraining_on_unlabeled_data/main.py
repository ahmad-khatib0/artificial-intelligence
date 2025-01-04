import torch
import tiktoken
import time
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
from config import GPT_CONFIG_124M

from importlib.metadata import version
from ch04_code import GPTModel, generate_text_simple
from ch02_code import create_dataloader_v1
from gpt_download import download_and_load_gpt2, load_downloaded_gpt2

pkgs = [
    "matplotlib",
    "numpy",
    "tiktoken",
    "torch",
    "tensorflow",  # For OpenAI's pretrained weights
]

for p in pkgs:
    print(f"{p} version: {version(p)}")


torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()


# Figure 5.3
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"],
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
# Every effort moves you rentingetic wasn? refres RexMeCHicular stren

inputs = torch.tensor(
    [[16833, 3626, 6100], [40, 1107, 588]]
)  # ["every effort moves", "I really like"]

# Matching these inputs, the targets contain the token IDs we want the model to produce:
targets = torch.tensor(
    [
        [3626, 6100, 345],  # [" effort moves you",
        [1107, 588, 11311],  # " really like chocolate"]
    ]
)

with torch.no_grad():
    logits = model(inputs)
probas = torch.softmax(logits, dim=-1)  # Probability of each token in vocabulary
print(probas.shape)
# torch.Size([2, 3, 50257]) => batch size, number of tokensin each input, embedding dimensionality

token_ids = torch.argmax(probas, dim=-1, keepdim=True)
print("Token IDs:\n", token_ids)
# Token IDs: tensor([[[16657], [339], [42826]], [[49906], [29669], [41751]]])
#                         first batch                 second batch

print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Outputs batch 1:" f" {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")
# Targets batch 1: effort moves you
# Outputs batch 1: Armed heNetflix


# For each of the two input texts, we can print the initial softmax
# probability scores corresponding to the target tokens
text_idx = 0
target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 1:", target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
print("Text 2:", target_probas_2)
# Text 1: tensor([7.4541e-05, 3.1061e-05, 1.1563e-05])
# Text 2: tensor([1.0337e-05, 5.6776e-05, 4.7559e-06])


# applying the logarithm to the probability scores:
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)

# Next, we combine these log probas into a single score by computing the average (step 5 in figure 5.7):
avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)
# tensor(-10.7940)

# The goal is to get the average log probability as close to 0 as possible by updating the model’s
# weights as part of the training process. However, in deep learning, the common practice isn’t to
# push the average log probability up to 0 but rather to bring the negative average log probability
# down to 0. The negative average log probability is simply the average log probability multiplied
# by –1, which corresponds to step 6 in figure 5.7:
neg_avg_log_probas = avg_log_probas * -1
print(neg_avg_log_probas)
# tensor(10.7940)

print("Logits shape:", logits.shape)
print("Targets shape:", targets.shape)
# Logits shape: torch.Size([2, 3, 50257])
# Targets shape: torch.Size([2, 3])


# For the cross_entropy loss function in PyTorch, we want to flatten these tensors
# by combining them over the batch dimension:
logits_flat = logits.flatten(0, 1)
targets_flat = targets.flatten()
print("Flattened logits:", logits_flat.shape)  #   torch.Size([6, 50257])
print("Flattened targets:", targets_flat.shape)  # torch.Size([6])


# Previously, we applied the softmax function, selected the probability scores
# corresponding to the target IDs, and computed the negative average log probabilities.
# PyTorch’s cross_entropy function will take care of all these steps for us:
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)

print(loss)  # tensor(10.7940)
# The resulting loss is the same that we obtained previously when
# applying the individual steps in figure 5.7 manually:

file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()


total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)  # 20479
print("Tokens:", total_tokens)  # 5145


train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]  # the remaining 10 percentage
print("split idx is: ", split_idx)


torch.manual_seed(123)
train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0,
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0,
)


# optional check, we can iterate through the data loaders to ensure that they were created correctly:
print("Train Loader:")
for x, y in train_loader:
    print(x.shape, y.shape)

print("\nValidation loader:")
for x, y in val_loader:
    print(x.shape, y.shape)

# Train loader:
# torch.Size([2, 256]) torch.Size([2, 256])
# torch.Size([2, 256]) torch.Size([2, 256])
# torch.Size([2, 256]) torch.Size([2, 256])
# torch.Size([2, 256]) torch.Size([2, 256])
# torch.Size([2, 256]) torch.Size([2, 256])
# torch.Size([2, 256]) torch.Size([2, 256])
# torch.Size([2, 256]) torch.Size([2, 256])
# torch.Size([2, 256]) torch.Size([2, 256])
# torch.Size([2, 256]) torch.Size([2, 256])

# Validation loader:
# torch.Size([2, 256]) torch.Size([2, 256])

# we have nine training set batches with two samples and 256 tokens each. Since we allocated only 10%
# of the data for validation, there is only one validation batch consisting of two input examples.
# As expected, the input data (x) and target data (y) have the same shape (the batch size times the
# number of tokens in each batch) since the targets are the inputs shifted by one position.


def calc_loss_batch(input_batch, target_batch, model, device):
    """calculate the cross entropy loss of a given batch returned via the training and validation loader"""
    # The transfer to a given device allows us to transfer the data to a GPU.
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    "Compute the training and validation loss"
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        # Iteratives over all batches if no fixed num_batches is specified
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()  # sums loss for each batch
        else:
            break
    return total_loss / num_batches  # Averages the loss over all batches


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# a CUDA-supported GPU?, the LLM will train on the GPU without making any changes to the code.
model.to(device)

# Disables gradient tracking for efficiency because we are not training yet
with torch.no_grad():
    # Via the “device” setting we ensure the data is loaded onto the same device as the LLM model.
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)
print("Training loss:", train_loss)  # Training loss: 10.98758347829183
print("Validation loss:", val_loss)  # Validation loss: 10.98110580444336
# The loss values are relatively high because the model has not yet been trained.


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
):
    "The main function for pretraining LLMs"

    # Initializes lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        # Print a sample text after each epoch
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    # Dropout is disabled during evaluation for stable, reproducible results.
    model.eval()
    # Disables gradient tracking, which is not required during evaluation, to reduce the computational overhead
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)

    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


start_time = time.time()

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
# The .parameters() method returns all trainable weight parameters of the model.
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs=num_epochs,
    eval_freq=5,
    eval_iter=5,
    start_context="Every effort moves you",
    tokenizer=tokenizer,
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(
        MaxNLocator(integer=True)
    )  # only show integer labels on x-axis

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig("loss-plot.pdf")
    plt.show()


epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

# Decoding strategies (text generation) to control randomness (to reduce training data memorization)

# inference with a relatively small model does not require a GPU
model.to("cpu")
#  evaluation mode to turn off random components such as dropout
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids("Evert effort moves you", tokenizer),
    max_new_tokens=25,
    context_size=GPT_CONFIG_124M["context_length"],
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
# Every effort moves you know," was one of the axioms he laid down across the
# Sevres and silver of an exquisitely appointed lun


def generate(
    model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None
):
    """
    combine temperature sampling and top-k sampling to modify the generate_text_simple
    function we used to generate text via the LLM earlier
    """
    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits
            )

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        # Stop generating early if end-of-sequence token is encountered and eos_id is specified
        if idx_next == eos_id:
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx


torch.manual_seed(123)
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4,
)
print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

# Every effort moves you stand to work on surprise, a one of us had gone with random
#
# As we can see, the generated text is very different from the one we previously generated via
# the generate_simple function in section 5.3 ("Every effort moves you know," was one of the
# axioms he laid...! ),  which was a  memorized passage from the training set.

# Loading and saving model weights in PyTorch
torch.save(model.state_dict(), "model.pth")


model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# during inference, we don’t want to randomly drop out any of the information the network has learned.
# Using model.eval() switches the model to evaluation mode for inference, disabling the dropout layers of
# the model. If we plan to continue pretraining a model later—for example, using the train_model_simple
# function we defined earlier in this chapter—saving the optimizer state is also recommended.


torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
    "model_and_optimizer.pth",
)

# restore the model and optimizer states
checkpoint = torch.load("model_and_optimizer.pth", map_location=device)
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.train()

# settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
