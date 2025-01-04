import re
import torch
from torch.utils.data import Dataset, DataLoader
from importlib.metadata import version
import tiktoken

# url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
# file_path = "the-verdict.txt"
# urllib.request.urlretrieve(url, file_path)

print("tiktoken version", version("tiktoken"))
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))
# 4690, which is the number of tokens in this text (without whitespaces).

# splits on whitespaces (\s), commas, and periods ([,.]), quotation marks, and the double-dashes
text = "Hello, world. This, is a test."
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)

result = [item for item in result if item.strip()]
print(result)


all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)  # 1,130

vocab = {token: integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break

# ('!', 0) ... ('Hermia', 50)
# apply this vocabulary to convert new text into token IDs


class SimpleTokenizerV1:
    def __init__(self, vocab) -> None:
        # Stores the vocabulary as a class attribute for access in the encode and decode methods
        self.str_to_int = vocab
        # Creates an inverse vocabulary that maps token IDs back to the original text tokens
        self.int_to_str = {i: s for s, i in vocab.items()}

    # processes input text into token ids
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    # Converts token IDs back into text
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Removes spaces before the specified punctuation
        text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
        return text


tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know,"
Mrs. Gisburn said with pardonable pride."""

ids = tokenizer.encode(text)
print(ids)

print(tokenizer.decode(ids))

# text = "Hello, do you like tea?"
# print(tokenizer.encode(text)) #
# Executing this code will result in the following error:       KeyError: 'Hello'
# The problem is that the word “Hello” was not used in the “The Verdict” short story.

#
# We need to modify the tokenizer to handle unknown words, We need to modify the tokenizer to handle
# unknown words. We also need to address the usage and addition of special context tokens that can
# enhance a model’s understanding of context or other relevant information in the text. These special
# tokens can include markers for unknown words and document boundaries,
# For instance, we add an <|unk|> token to represent new and unknown words that were not part of the
# training data and thus not part of the existing vocabulary. Furthermore, we add an <|endoftext|>
# token that we can use to separate two unrelated text sources.

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}

print(len(vocab.items()))  #  the new vocabulary size is 1,132 (previous was 1,130)
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)


class SimpleTokenizerV2:
    def __init__(self, vocab) -> None:
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        # replace unknown words by <|unk|> tokens
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>" for item in preprocessed
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replaces spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r"\1", text)
        return text


text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)


tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(text))
# [1131, 5, 355, 1126, 628, 975, 10, 1130, 55, 988, 956, 984, 722, 988, 1131, 7]
# We can see that the list of token IDs contains 1130 for the <|endoftext|> separator
# token as well as two 1131 tokens, which are used for unknown words.

print(tokenizer.decode(tokenizer.encode(text)))
# <<|unk|>, do you like tea? <|endoftext|> In the sunlit terraces of the <|unk|>.


tokenizer = tiktoken.get_encoding("gpt2")
text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
strings = tokenizer.decode(integers)
print(strings)


# Data sampling with a sliding window

# Let’s implement a data loader that fetches the input–target pairs in
# figure 2.12 from the training dataset using a sliding window approach
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))  # 5145, the total number of tokens in the training set,

enc_sample = enc_text[:50]

# The context size determines how many tokens are included in the input
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1 : context_size + 1]
print(f"x: {x}")
print(f"y:      {y}")
# x: [290, 4920, 2241, 287]
# y:       [4920, 2241, 287, 257]

# By processing the inputs along with the targets, which are the inputs shifted by one
# position, we can create the next-word prediction tasks (see figure 2.12), as follows:
for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)

# [290] ----> 4920
# [290, 4920] ----> 2241
# [290, 4920, 2241] ----> 287
# [290, 4920, 2241, 287] ----> 257

for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "--->", tokenizer.decode([desired]))

# I --->  H
# I H ---> AD
# I HAD --->  always
# I HAD always --->  thought


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt)  # Tokenizes the entire text
        # Uses a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    # Returns the total number of rows in the dataset
    def __len__(self):
        return len(self.input_ids)

    # Returns a single row from the dataset
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]


# load the inputs in batches via a PyTorch DataLoader.
def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,  #  to prevent loss spikes during training.
        num_workers=num_workers,  # The number of CPU processes to use for preprocessing
    )

    return dataloader


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)

# Converts dataloader into a Python iterator to fetch the next entry via Python’s built-in next() function
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
# [tensor([[ 40, 367, 2885, 1464]]), tensor([[ 367, 2885, 1464, 1807]])]

second_batch = next(data_iter)
print(second_batch)
# [tensor([[ 367, 2885, 1464, 1807]]), tensor([[2885, 1464, 1807, 3619]])]

# If we compare the first and second batches, we can see that the second batch’s token
# IDs are shifted by one position (for example, the second ID in the first batch’s input is
# 367, which is the first ID of the second batch’s input). The stride setting dictates the
# number of positions the inputs shift across batches, emulating a sliding window approach,


dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n", targets)


# Creating token embeddings (example)
input_ids = torch.tensor([2, 3, 5, 1])
#  suppose we have a small vocabulary of only 6 words (instead of the 50,257 words in the BPE tokenizer
# vocabulary), and we want to create embeddings of size 3 (in GPT-3, the embedding size is 12,288 dimensions):
vocab_size = 6
output_dim = 3

# now we can instantiate an embedding layer in PyTorch
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)
# tensor([
#     [ 0.3374, -0.1778, -0.1690],
#     [ 0.9178, 1.5810, 1.3010],
#     [ 1.2753, -0.2010, -0.1606],
#     [-0.4015, 0.9666, -1.1481],
#     [-1.1589, 0.3255, -0.6315],
#     [-2.8400, -0.7849, -1.4096]
# ], requires_grad=True)
#
# The weight matrix of the embedding layer contains small, random values. These values are optimized
# during LLM training as part of the LLM optimization itself. More- over, we can see that the weight
# matrix has six rows and three columns. There is one row for each of the six possible tokens in the
# vocabulary, and there is one column for each of the three embedding dimensions.

# let’s apply it to a token ID to obtain the embedding vector:
print(embedding_layer(torch.tensor([3])))
# tensor([[-0.4015, 0.9666, -1.1481]], grad_fn=<EmbeddingBackward0>)
#
# If we compare the embedding vector for token ID 3 to the previous embedding matrix, we see that it
# # is identical to the fourth row (Python starts with a zero index, so it’s the row corresponding to
# index 3). In other words, the embedding layer is essen- tially a lookup operation that retrieves
# rows from the embedding layer’s weight matrix via a token ID.

print(embedding_layer(input_ids))  # 4 × 3 matrix:
# tensor([
#     [ 1.2753, -0.2010, -0.1606],
#     [-0.4015, 0.9666, -1.1481],
#     [-2.8400, -0.7849, -1.4096],
#     [ 0.9178, 1.5810, 1.3010]], grad_fn=<EmbeddingBackward0>)

# Encoding word positions

vocab_size = 50257  # BPE which has a vocabulary size of 50,257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# 8 × 4 × 256 tensor
max_length = 4
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)
# torch.Size([8,4]):  consists of eight text samples with four tokens each

# embed these token IDs into 256-dimensional vectors:
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)  # torch.Size([8, 4, 256])
# 8 × 4 × 256–dimensional tensor shows that each token ID is now embedded as a 256-dimensional vector.


# For a GPT model’s absolute embedding approach, we just need to create another
# embedding layer that has the same embedding dimension as the token_embedding_layer :
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)  # torch.Size([4, 256])

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)  # torch.Size([8, 4, 256])
