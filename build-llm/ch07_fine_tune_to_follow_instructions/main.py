import json
import time
from functools import partial
from tensorflow.python.framework.ops import re
from tqdm import tqdm
import tiktoken
import psutil
import urllib.request

import torch
from torch.utils.data import Dataset, DataLoader

from dataset_download import download_and_load_file
from gpt_download import download_and_load_gpt2
from config import get_config
from ch04_code import GPTModel
from ch05_code import (
    load_weights_into_gpt,
    generate,
    text_to_token_ids,
    token_ids_to_text,
    calc_loss_loader,
    train_model_simple,
    plot_losses,
)


def format_input(entry):
    "Convert the entries in the data list into the Alpaca-style input format"
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


def test_format_input(data):
    model_input = format_input(data[50])
    desired_response = f"\n\n### Response:\n{data[50]['output']}"
    print(model_input + desired_response)

    # Below is an instruction that describes a task. Write a response that
    # appropriately completes the request.
    #
    # ### Instruction:
    # Identify the correct spelling of the following word.
    #
    # ### Input:
    # Ocassion
    #
    # ### Response:
    # The correct spelling is 'Occasion.'

    # sometimes the input json field:
    model_input = format_input(data[999])
    desired_response = f"\n\n### Response:\n{data[999]['output']}"
    print(model_input + desired_response)

    # Below is an instruction that describes a task. Write a response that
    # appropriately completes the request.
    #
    ### Instruction:
    # What is an antonym of 'complicated'?
    #
    ### Response:
    # An antonym of 'complicated' is 'simple'.


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer) -> None:
        super().__init__()
        self.data = data
        self.encoded_texts = []
        for entry in data:
            # Pretokenizes texts
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def custom_collate_draft_1(batch, pad_token_id=50256, device="cpu"):
    """
    Moving on to step 2.3 of the process (see figure 7.6), we adopt a more sophisticated approach by
    developing a custom collate function that we can pass to the data loader. This custom collate
    function pads the training examples in each batch to the same length while allowing different
    batches to have different lengths, as demonstrated in figure 7.8. This approach minimizes
    unnecessary padding by only extending sequences to match the longest one in each batch,
    not the whole dataset.
    """

    # Find the longest sequence in the batch
    # and increase the max length by +1, which will add one extra
    # padding token below
    batch_max_length = max(len(item) + 1 for item in batch)

    # Pad and prepare inputs
    inputs_lst = []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to batch_max_length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        # Via padded[:-1], we remove the extra padded token
        # that has been added via the +1 setting in batch_max_length
        # (the extra padding token will be relevant in later codes)
        inputs = torch.tensor(padded[:-1])
        inputs_lst.append(inputs)

    # Convert list of inputs to tensor and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    return inputs_tensor


def get_testing_batch():
    inputs_1 = [0, 1, 2, 3, 4]
    inputs_2 = [5, 6]
    inputs_3 = [7, 8, 9]
    batch = (inputs_1, inputs_2, inputs_3)
    return batch


def custom_collate_draft_2(batch, pad_token_id=50256, device="cpu"):
    "The following updated collate function generates the target token IDs from the input token IDs"
    # Find the longest sequence in the batch
    batch_max_length = max(len(item) + 1 for item in batch)

    # Pad and prepare inputs
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs to tensor and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


def custom_collate_fn(
    batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"
):
    """
    note that we retain one end-of-text token, ID 50256, in the target list, as depicted in figure
    7.12. Retaining it allows the LLM to learn when to generate an end-of-text token in response to
    instructions, which we use as an indicator that the generated response is complete.
    """

    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []
    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

        # Replaces all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # New: Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


def explain_loss_affect():
    # For demonstration purposes, consider the following simple and self-contained example where each
    # output logit corresponds to a potential token from the model’s vocabulary. Here’s how we might
    # calculate the cross entropy loss (introduced in chapter 5) during training when the model predicts
    # a sequence of tokens, which is similar to what we did when we pretrained the model and fine-tuned
    # it for classification:

    logits_1 = torch.tensor(
        [
            [-1.0, 1.0],  # predictions for 1st token
            [-0.5, 1.5],  # predictions for 2nd token
        ]
    )

    targets_1 = torch.tensor([0, 1])  # Correct token indices to generate
    loss_1 = torch.nn.functional.cross_entropy(logits_1, targets_1)
    print(loss_1)  # tensor(1.1269)

    # As we would expect, adding an additional token ID affects the loss calculation:
    logits_2 = torch.tensor(
        [
            [-1.0, 1.0],
            [-0.5, 1.5],
            [-0.5, 1.5],  # New third token ID prediction
        ]
    )

    targets_2 = torch.tensor([0, 1, 1])
    loss_2 = torch.nn.functional.cross_entropy(logits_2, targets_2)
    print(loss_2)  # 0.7936

    # see what happens if we replace the third target token ID with -100 :
    targets_3 = torch.tensor([0, 1, -100])
    loss_3 = torch.nn.functional.cross_entropy(logits_2, targets_3)
    print(loss_3)
    print("loss_1 == loss_3:", loss_1 == loss_3)
    # tensor(1.1269)
    # loss_1 == loss_3: tensor(True)

    # In other words, the cross entropy loss function ignored the third entry in the targets_3 vector,
    # the token ID corresponding to -100. (Interested readers can try to replace the -100 value with
    # another token ID that is not 0 or 1; it will result in an error.)
    #
    # So what’s so special about -100 that it’s ignored by the cross entropy loss? The default setting
    # of the cross entropy function in PyTorch is cross_entropy(..., ignore_index=-100). This means
    # that it ignores targets labeled with -100. We take advantage of this ignore_index to ignore the
    # additional end-of-text (padding) tokens that we used to pad the training examples to have the
    # same length in each batch. However, we want to keep one 50256 (end-of-text) token ID in the
    # targets because it helps the LLM to learn to generate end-of-text tokens, which we can use as
    # an indicator that a response is complete.


def check_if_running(process_name):
    """
    verifies that the Ollama session is running properly before we use Ollama to evaluate
    the test set responses
    """
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running


def query_model(prompt, model="llama3", url="http://localhost:11434/api/chat"):
    # Create the data payload as a dictionary
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "options": {  # Settings below are required for deterministic responses
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048,
        },
    }

    # Convert the dictionary to a JSON formatted string and encode it to bytes
    payload = json.dumps(data).encode("utf-8")

    # Create a request object, setting the method to POST and adding necessary headers
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    # Send the request and capture the response
    response_data = ""
    with urllib.request.urlopen(request) as response:
        # Read and decode the response
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

    return response_data


def evaluate_model_using_llama_v1(test_data):
    for entry in test_data[:3]:
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry['model_response']}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
        )

        print("\nDataset response:")
        print(">>", entry["output"])
        print("\nModel response:")
        print(">>", entry["model_response"])
        print("\nScore:")
        print(">>", query_model(prompt))
        print("\n-------------------------")

    # Dataset response:
    # >> The car is as fast as lightning.

    # Model response:
    # >> The car is as fast as a bullet.

    # Score:
    # >> I'd rate the model response "The car is as fast as a bullet." an 85 out of 100.

    # Here's why:

    # * The response uses a simile correctly, comparing the speed of the car to something else (in this case, a bullet).
    # * The comparison is relevant and makes sense, as bullets are known for their high velocity.
    # * The phrase "as fast as" is used correctly to introduce the simile.

    # The only reason I wouldn't give it a perfect score is that some people might find the comparison slightly less vivid or evocative than others. For example, comparing something to lightning (as in the original response) can be more dramatic and attention-grabbing. However, "as fast as a bullet" is still a strong and effective simile that effectively conveys the idea of the car's speed.

    # Overall, I think the model did a great job!

    # -------------------------

    # Dataset response:
    # >> The type of cloud typically associated with thunderstorms is cumulonimbus.

    # Model response:
    # >> The type of cloud associated with thunderstorms is a cumulus cloud.

    # Score:
    # >> I'd score this model response as 40 out of 100.

    # Here's why:

    # * The model correctly identifies that thunderstorms are related to clouds (correctly identifying the type of phenomenon).
    # * However, it incorrectly specifies the type of cloud associated with thunderstorms. Cumulus clouds are not typically associated with thunderstorms; cumulonimbus clouds are.
    # * The response lacks precision and accuracy in its description.

    # Overall, while the model attempts to address the instruction, it provides an incorrect answer, which is a significant error.

    # -------------------------

    # Dataset response:
    # >> Jane Austen.

    # Model response:
    # >> The author of 'Pride and Prejudice' is Jane Austen.

    # Score:
    # >> I'd rate my own response as 95 out of 100. Here's why:

    # * The response accurately answers the question by naming the author of 'Pride and Prejudice' as Jane Austen.
    # * The response is concise and clear, making it easy to understand.
    # * There are no grammatical errors or ambiguities that could lead to confusion.

    # The only reason I wouldn't give myself a perfect score is that the response is slightly redundant - it's not necessary to rephrase the question in the answer. A more concise response would be simply "Jane Austen."

    # -------------------------


def generate_model_scores(json_data, json_key, model="llama3"):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        # Modified instructions line to only return the score
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"Respond with the integer number only."
        )

        score = query_model(prompt, model)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score: {score}")
            continue

    return scores


if __name__ == "__main__":
    file_path = "instruction-data.json"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"

    data = download_and_load_file(file_path, url)
    print("Number of entries:", len(data))

    test_format_input(data)

    train_portion = int(len(data) * 0.85)  # 85% for training
    test_portion = int(len(data) * 0.1)  # 10% for testing
    val_portion = (
        len(data) - train_portion - test_portion
    )  # Remaining 5% for validation

    train_data = data[:train_portion]
    test_data = data[train_portion : train_portion + test_portion]
    val_data = data[train_portion + test_portion :]

    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))
    # Training set length: 935
    # Validation set length: 55
    # Test set length: 110

    batch = get_testing_batch()
    print(custom_collate_draft_1(batch))
    # tensor([[0, 1, 2, 3, 4], [5, 6, 50256, 50256, 50256], [7, 8, 9, 50256, 50256]])
    #
    # This output shows all inputs have been padded to the length of the
    # longest input list, inputs_1, containing five token IDs.

    inputs, targets = custom_collate_draft_2(batch)
    print(inputs), print(targets)

    # The first tensor represents inputs.
    # tensor([[0, 1, 2, 3, 4], [5, 6, 50256, 50256, 50256], [7, 8, 9, 50256, 50256]])
    # The second tensor represents the targets.
    # tensor(
    #     [
    #         [1, 2, 3, 4, 50256],
    #         [6, 50256, 50256, 50256, 50256],
    #         [8, 9, 50256, 50256, 50256],
    #     ]
    # )

    inputs, targets = custom_collate_fn(batch)
    print(inputs), print(targets)

    # tensor([[0, 1, 2, 3, 4], [5, 6, 50256, 50256, 50256], [7, 8, 9, 50256, 50256]])
    # tensor(
    # #     first pos is the second el in the prev tensor
    #     [[1, 2, 3, 4, 50256], [6, 50256, -100, -100, -100], [8, 9, 50256, -100, -100]]
    # )

    # What is the logic behind this adjustment (why using -100)?
    explain_loss_affect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    customized_collate_fn = partial(
        custom_collate_fn, device=device, allowed_max_length=1024
    )

    # try to increase this number if parallel Python processes are supported by your os
    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)

    tokenizer = tiktoken.get_encoding("gpt2")
    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    print("Train loader:")
    for inputs, targets in train_loader:
        print(inputs.shape, targets.shape)
    # Train loader:
    # torch.Size([8, 61]) torch.Size([8, 61])  # batch size, number of tokens
    # torch.Size([8, 76]) torch.Size([8, 76])
    # torch.Size([8, 73]) torch.Size([8, 73])

    config, model_size, CHOOSE_MODEL = get_config()
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
    model = GPTModel(config)
    load_weights_into_gpt(model, params)
    model.eval()

    # take a moment to assess the pretrained LLM’s performance on one of the validation tasks
    torch.manual_seed(123)
    input_text = format_input(val_data[0])

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer),
        max_new_tokens=35,
        context_size=config["context_length"],
        eos_id=50256,
    )

    generated_text = token_ids_to_text(token_ids, tokenizer)

    # when evaluating the model’s performance on a specific task, we often want to focus solely on
    # the model’s generated response. To isolate the model’s response text, we need to subtract the
    # length of the input instruction from the start of the generated_text:
    response_text = generated_text[len(input_text) :].strip()
    print(response_text)

    # Fine-tuning the LLM on instruction data
    # Before we begin training, let’s calculate the initial loss for the training and validation sets:
    model.to(device)
    torch.manual_seed(123)

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
        print("Training loss:", train_loss)
        print("Validation loss:", val_loss)
    # Training loss: 3.825908660888672           Validation loss: 3.7619335651397705

    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
    num_epochs = 2

    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context=format_input(val_data[0]),
        tokenizer=tokenizer,
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    torch.manual_seed(123)
    # Iterates over the first three test set samples
    for entry in test_data[:3]:
        input_text = format_input(entry)
        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=config["context_length"],
            eos_id=50256,
        )

        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = (
            generated_text[len(input_text) :].replace("### Response:", "").strip()
        )

        print(input_text)
        print(f"\nCorrect response:\n>> {entry['output']}")
        print(f"\nModel response:\n>> {response_text.strip()}")
        print("-------------------------------------")

    # Generating test set responses (using another LLM to evaluate our fine-tuned model’s responses)
    #
    # tqdm: Instantly make your loops show a smart progress meter
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        input_text = format_input(entry)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=config["context_length"],
            eos_id=50256,
        )

        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = (
            generated_text[len(input_text) :].replace("### Response:", "").strip()
        )

        test_data[i]["model_response"] = response_text

    with open("instruction-data-with-response.json", "w") as file:
        json.dump(test_data, file, indent=4)

    print(test_data[0])
    # {
    #     "instruction": "Rewrite the sentence using a simile.",
    #     "input": "The car is very fast.",
    #     "output": "The car is as fast as lightning.",
    #     "model_response": "The car is as fast as a bullet.",
    # }

    file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft.pth"
    torch.save(model.state_dict(), file_name)
    print(f"Model saved as {file_name}")

    # $$ ollama serve
    # $$ ollama run llama3

    ollama_running = check_if_running("ollama")
    if not ollama_running:
        raise RuntimeError("Ollama not running. Launch ollama before proceeding.")
    print("Ollama running:", check_if_running("ollama"))

    model = "llama3"
    result = query_model("What do Llamas eat?", model)
    print(result)

    # Using the query_model function defined earlier, we can evaluate the responses generated
    # by our fine-tuned model that prompts the Llama 3 model to rate our fine- tuned model’s
    # responses on a scale from 0 to 100 based on the given test set response as reference.
    evaluate_model_using_llama_v1(test_data)

    scores = generate_model_scores(test_data, "model_response")
    print(f"Number of scores: {len(scores)} of {len(test_data)}")
    print(f"Average score: {sum(scores)/len(scores):.2f}\n")
    # Number of scores: 110 of 110
    # Average score: 50.32
