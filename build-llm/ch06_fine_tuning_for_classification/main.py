import time
import torch
import tiktoken
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, Dataset

from dataset_download import data_file_path
from config import BASE_CONFIG, CHOOSE_MODEL
from ch04_code import generate_text_simple
from ch05_code import (
    GPTModel,
    load_downloaded_gpt2,
    load_weights_into_gpt,
    text_to_token_ids,
    token_ids_to_text,
)


def create_balanced_dataset(df):
    # Count the instances of "spam"
    num_spam = df[df["Label"] == "spam"].shape[0]

    # Randomly sample "ham" instances to match the number of "spam" instances
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)

    # Combine ham "subset" with "spam"
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    return balanced_df


def random_split(df, train_frac, validation_frac):
    """
    Split the dataset into three parts: 70% for training, 10% for validation, and 20% for testing.
    (These ratios are common in machine learning to train, adjust, and evaluate models.)
    """

    # Shuffle the entire DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Calculate split indices
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


def split_and_save_dataset():
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])

    print(df["Label"].value_counts())
    # ham 4825 spam 747 dtype: int64
    balanced_df = create_balanced_dataset(df)
    print(balanced_df["Label"].value_counts())
    # ham     747    spam    747 Name: count, dtype: int64

    # convert the “string” class labels "ham" and "spam" into integer class labels 0 and 1, respectively:
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    # Test size is implied to be 0.2 as the remainder.
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    train_df.to_csv("train.csv", index=None)
    validation_df.to_csv("validation.csv", index=None)
    test_df.to_csv("test.csv", index=None)


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        # Pre-tokenize texts
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # Truncate sequences if they are longer than max_length
            self.encoded_texts = [
                encoded_text[: self.max_length] for encoded_text in self.encoded_texts
            ]

        # Pad sequences to the longest sequence
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


def create_data_loaders(train_dataset, val_dataset, test_dataset):
    num_workers = 0  # This setting ensures compatibility with most computers
    batch_size = 8
    torch.manual_seed(123)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    for input_batch, target_batch in train_loader:
        pass

    print("Input batch dimensions:", input_batch.shape)
    print("Label batch dimensions", target_batch.shape)
    # Input batch dimensions: torch.Size([8, 120])
    # Label batch dimensions torch.Size([8])
    # As we can see, the input batches consist of eight training examples with 120 tokens each, as
    # expected. The label tensor stores the class labels corresponding to the eight training examples.

    print(f"{len(train_loader)} training batches")
    print(f"{len(val_loader)} validation batches")
    print(f"{len(test_loader)} test batches")
    # 130 training batches
    # 19 validation batches
    # 38 test batches

    return train_loader, val_loader, test_loader


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    """
    To determine the classification accuracy, we apply the argmax-based prediction
    code to all examples in the dataset and calculate the proportion of correct predictions
    """
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i == 2 or i == 3:
            print(input_batch)
            print(target_batch)
            print(enumerate(data_loader))
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]  # Logits of last output token
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples


def calc_loss_batch(input_batch, target_batch, model, device):
    """
    Because classification accuracy is not a differentiable function, we use cross-entropy loss as
    a proxy to maximize accuracy. Accordingly, the calc_loss_batch function remains the same, with
    one adjustment: we focus on optimizing only the last token, model(input_batch)[:, -1, :],
    rather than all tokens, model(input_batch)
    """

    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    # Logits of last output token
    logits = model(input_batch)[:, -1, :]

    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """To calculate the loss for all batches in a data loader"""

    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")

    elif num_batches is None:
        num_batches = len(data_loader)
    # Ensures number of batches doesn’t exceed batches in data loader
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def train_classifier_simple(
    model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter
):
    """
    The training function implementing the concepts shown in figure 6.15 also closely mirrors the
    train_model_simple function used for pretraining the model. The only two distinctions are that
    we now track the number of training examples seen (examples_seen) instead of the number of
    tokens, and we calculate the accuracy after each epoch instead of printing a sample text
    """

    # Initialize lists to track losses and examples seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            examples_seen += input_batch.shape[
                0
            ]  # New: track examples instead of tokens
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        # Calculate accuracy after each epoch
        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter
        )
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    "The evaluate_model function is identical to the one we used for pretraining"
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)

    model.train()
    return train_loss, val_loss


def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # Create a second x-axis for examples seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(examples_seen, train_values, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig(f"{label}-plot.pdf")
    plt.show()


def classify_review(
    text, model, tokenizer, device, max_length=None, pad_token_id=50256
):
    model.eval()

    # Prepare inputs to the model
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]
    # Note: In the book, this was originally written as pos_emb.weight.shape[1] by mistake
    # It didn't break the code but would have caused unnecessary truncation (to 768 instead of 1024)

    # Truncate sequences if they too long
    input_ids = input_ids[: min(max_length, supported_context_length)]

    # Pad sequences to the longest sequence
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    # add batch dimension
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    # Model inference
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]  # Logits of the last output token
    predicted_label = torch.argmax(logits, dim=-1).item()

    # Return the classified result
    return "spam" if predicted_label == 1 else "not spam"


# spam and not spam (ham) classifier
if __name__ == "__main__":
    # split_and_save_dataset()
    tokenizer = tiktoken.get_encoding("gpt2")
    # print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))  # [50256]

    train_dataset = SpamDataset(
        csv_file="train.csv", max_length=None, tokenizer=tokenizer
    )
    print(train_dataset.max_length)  # 120 tokens
    # The model can handle sequences of up to 1,024 tokens, given its context length limit. If your
    # dataset includes longer texts, you can pass max_length=1024 when creating the training dataset
    # to ensure that the data does not exceed the model’s supported input (context) length

    val_dataset = SpamDataset(
        csv_file="validation.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer,
    )
    test_dataset = SpamDataset(
        csv_file="test.csv", max_length=train_dataset.max_length, tokenizer=tokenizer
    )

    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=val_dataset
    )

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = load_downloaded_gpt2(model_size=model_size, models_dir="gpt2")

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()

    text_1 = "Every effort moves you"
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(text_1, tokenizer),
        max_new_tokens=15,
        context_size=BASE_CONFIG["context_length"],
    )
    print(token_ids_to_text(token_ids, tokenizer))
    # Every effort moves you forward.
    # The first step is to understand the importance of your work

    text_2 = (
        "Is the following text 'spam'? Answer with 'yes' or 'no':"
        " 'You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award.'"
    )

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(text_2, tokenizer),
        max_new_tokens=23,
        context_size=BASE_CONFIG["context_length"],
    )
    print(token_ids_to_text(token_ids, tokenizer))

    # To get the model ready for classification fine-tuning, we first freeze the model,
    # meaning that we make all layers nontrainable:
    for param in model.parameters():
        param.requires_grad = False

    torch.manual_seed(123)
    num_classes = 2
    model.out_head = torch.nn.Linear(
        in_features=BASE_CONFIG["emb_dim"], out_features=num_classes
    )

    # make the final LayerNorm and last transformer block trainable
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True

    # Even though we added a new output layer and marked certain layers as trainable or nontrainable,
    # we can still use this model similarly to how we have previously
    inputs = tokenizer.encode("Do you have time")
    inputs = torch.tensor(inputs).unsqueeze(0)
    print("Inputs:", inputs)
    print("Inputs dimensions:", inputs.shape)  # shape: (batch_size, num_tokens)
    # Inputs: tensor([[5211, 345, 423, 640]]) # 4 input tokens
    # Inputs dimensions: torch.Size([1, 4])

    with torch.no_grad():
        outputs = model(inputs)
    print("Outputs:\n", outputs)
    # tensor( [[[-1.5854, 0.9904], [-3.7235, 7.4548], [-2.2661, 6.6049], [-3.5983, 3.9902]]])
    print("Outputs dimensions:", outputs.shape)
    # Outputs dimensions: torch.Size([1, 4, 2])

    # Remember that we are interested in fine-tuning this model to return a class label indicating
    # whether a model input is “spam” or “not spam.” We don’t need to fine- tune all four output rows;
    # instead, we can focus on a single output token. In particu- lar, we will focus on the last row
    # corresponding to the last output token, as shown in figure 6.11.
    print("Last output token:", outputs[:, -1, :])
    # Last output token: tensor([[-3.5983, 3.9902]])
    #
    # Given the causal attention mask setup in figure 6.12, the last token in a sequence accumulates
    # the most information since it is the only token with access to data from all the previous tokens.
    # Therefore, in our spam classification task, we focus on this last token during the fine-tuning process.

    # We can obtain the class label:
    probas = torch.softmax(outputs[:, -1, :], dim=-1)
    label = torch.argmax(probas)
    print("Class label:", label.item())
    # 1,  meaning the model predicts that the input text is “spam.”

    #  Using the softmax function here is optional because the largest outputs directly correspond
    # to the highest probability scores. Hence, we can simplify the code without using softmax:
    logits = outputs[:, -1, :]
    label = torch.argmax(logits)
    print("Class label:", label.item())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    torch.manual_seed(123)

    train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
    val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
    test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)

    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")
    # Training accuracy: 46.25%
    # Validation accuracy: 45.00%
    # Test accuracy: 48.75%

    # Disables gradient tracking for efficiency because we are not training yet
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
        test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)

    print(f"Training loss: {train_loss:.3f}")
    print(f"Validation loss: {val_loss:.3f}")
    print(f"Test loss: {test_loss:.3f}")
    # Training loss: 2.453
    # Validation loss: 2.583
    # Test loss: 2.322

    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    num_epochs = 5

    train_losses, val_losses, train_accs, val_accs, examples_seen = (
        train_classifier_simple(
            model,
            train_loader,
            val_loader,
            optimizer,
            device,
            num_epochs=num_epochs,
            eval_freq=50,
            eval_iter=5,
        )
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    # Ep 1 (Step 000000): Train loss 2.153, Val loss 2.392
    # Ep 1 (Step 000050): Train loss 0.617, Val loss 0.637
    # Ep 1 (Step 000100): Train loss 0.523, Val loss 0.557 Training accuracy: 70.00% | Validation accuracy: 72.50%
    # Ep 2 (Step 000150): Train loss 0.561, Val loss 0.489
    # Ep 2 (Step 000200): Train loss 0.419, Val loss 0.397
    # Ep 2 (Step 000250): Train loss 0.409, Val loss 0.353 Training accuracy: 82.50% | Validation accuracy: 85.00%
    # Ep 3 (Step 000300): Train loss 0.333, Val loss 0.320
    # Ep 3 (Step 000350): Train loss 0.340, Val loss 0.306 Training accuracy: 90.00% | Validation accuracy: 90.00%
    # Ep 4 (Step 000400): Train loss 0.136, Val loss 0.200
    # Ep 4 (Step 000450): Train loss 0.153, Val loss 0.132
    # Ep 4 (Step 000500): Train loss 0.222, Val loss 0.137 Training accuracy: 100.00% | Validation accuracy: 97.50%
    # Ep 5 (Step 000550): Train loss 0.207, Val loss 0.143
    # Ep 5 (Step 000600): Train loss 0.083, Val loss 0.074 Training accuracy: 100.00% | Validation accuracy: 97.50%

    # plot the loss function for the training and validation set.
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
    plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

    # plot the classification accuracies:
    epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
    plot_values(
        epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy"
    )

    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_accuracy_loader(val_loader, model, device)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)

    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")

    text_1 = (
        "You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award."
    )
    print(
        classify_review(
            text_1, model, tokenizer, device, max_length=train_dataset.max_length
        )
    )  # "spam"

    text_2 = (
        "Hey, just wanted to check if we're still on"
        " for dinner tonight? Let me know!"
    )
    print(
        classify_review(
            text_2, model, tokenizer, device, max_length=train_dataset.max_length
        )
    )  # not spam

    torch.save(model.state_dict(), "review_classifier.pth")

    model_state_dict = torch.load("review_classifier.pth, map_location=device")
    model.load_state_dict(model_state_dict)
