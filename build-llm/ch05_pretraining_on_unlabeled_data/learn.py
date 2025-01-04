import torch
import matplotlib.pyplot as plt


f = torch.tensor([[[2, 4], [6, 8]], [[10, 12], [14, 16]]])
print(f.shape)  # torch.Size([2, 2, 2])
flattened = f.flatten()

print(flattened)  #       tensor([ 2,  4,  6,  8, 10, 12, 14, 16])
print(flattened.shape)  # torch.Size([8])

# To illustrate the probabilistic sampling with a concrete example, let’s briefly discuss the
# next-token generation process using a very small vocabulary for illustration purposes:
vocab = {
    "closer": 0,
    "every": 1,
    "effort": 2,
    "forward": 3,
    "inches": 4,
    "moves": 5,
    "pizza": 6,
    "toward": 7,
    "you": 8,
}

inverse_vocab = {v: k for k, v in vocab.items()}
# Next, assume the LLM is given the start context "every effort moves you"
# and generates the following next-token logits:
next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)

probas = torch.softmax(next_token_logits, dim=0)
next_token_id = torch.argmax(probas).item()
# print(inverse_vocab[next_token_id])

# To implement a probabilistic sampling process (to prevent the model from generating the same answer each time),
# we can now replace argmax with the multinomial function in PyTorch:
torch.manual_seed(123)
next_token_id = torch.multinomial(probas, num_samples=1).item()
print(next_token_id)
# print(inverse_vocab[next_token_id])


# The printed output is "forward" just like before. What happened? The multinomial function samples the next
# token proportional to its probability score. In other words, "forward" is still the most likely token and
# will be selected by multinomial most of the time but not all the time. To illustrate this, let’s implement
# a function that repeats this sampling 1,000 times:
def print_sampled_tokens(probas):
    torch.manual_seed(123)
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")


print_sampled_tokens(probas)
# 73 x closer
# 0 x every
# 0 x effort
# 582 x forward
# 2 x inches
# 0 x moves
# 0 x pizza
# 343 x toward

# As we can see, the word forward is sampled most of the time (582 out of 1,000 times),

# We can further control the distribution and selection process via a concept called temperature scaling.
# Temperature scaling is just a fancy description for dividing the logits by a number greater than 0:


def softmax_with_temperature(logits, temperature):
    print(logits)
    scaled_logits = logits / temperature
    print(scaled_logits)
    return torch.softmax(scaled_logits, dim=0)


# Temperatures greater than 1 result in more uniformly distributed token probabilities, and temperatures smaller
# than 1 will result in more confident (sharper or more peaky) distributions. Let’s illustrate this by plotting
# the original probabilities alongside probabilities scaled with different temperature values:
temperatures = [1, 0.1, 5]  # Original, lower, and higher confidence
scaled_probas = [softmax_with_temperature(next_token_logits, T) for T in temperatures]
x = torch.arange(len(vocab))
bar_width = 0.15

fig, ax = plt.subplots(figsize=(5, 3))
for i, T in enumerate(temperatures):
    rects = ax.bar(
        x + i * bar_width, scaled_probas[i], bar_width, label=f"Temperature = {T}"
    )

ax.set_ylabel("Probability")
ax.set_xticks(x)
ax.set_xticklabels(vocab.keys(), rotation=90)
ax.legend()
plt.tight_layout()
plt.show()

# A temperature of 1 divides the logits by 1 before passing them to the softmax function to compute the
# probability scores. In other words, using a temperature of 1 is the same as not using any temperature
# scaling. In this case, the tokens are selected with a probability equal to the original softmax probability
# scores via the multinomial sampling function in PyTorch. For example, for the temperature setting 1, the
# token corresponding to “forward” would be selected about 60% of the time, as we can see in figure 5.14.
# Also, as we can see in figure 5.14, applying very small temperatures, such as 0.1, will result in sharper
# distributions such that the behavior of the multinomial function selects the most likely token (here, "forward")
# almost 100% of the time, approaching the behavior of the argmax function. Likewise, a temperature of 5 results
# in a more uniform distribution where other tokens are selected more often. This can add more variety to the
# generated texts but also more often results in nonsensical text. For example, using the temperature of 5
# results in texts such as every effort moves you pizza about 4% of the time.


# In code, we can implement the top-k procedure in figure 5.15 as follows
top_k = 3
top_logits, top_pos = torch.topk(next_token_logits, top_k)
print("Top logits:", top_logits)
print("Top positions:", top_pos)
# Top logits: tensor([6.7500, 6.2800, 4.5100])
# Top positions: tensor([3, 7, 0])

# set the logit values of tokens that are below the lowest logit value within
# our top-three selection to negative infinity (-inf):
new_logits = torch.where(
    # Identifies logits less thanthe minimum in the top 3
    condition=next_token_logits < top_logits[-1],
    # Assigns –inf to these lower logits
    input=torch.tensor(float("-inf")),
    # Retains the original logits for all other tokens
    other=next_token_logits,
)

print(new_logits)
# tensor([4.5100, -inf, -inf, 6.7500, -inf, -inf, -inf, 6.2800, -inf])

# Lastly, let’s apply the softmax function to turn these into next-token probabilities:
topk_probas = torch.softmax(new_logits, dim=0)
print(topk_probas)

# tensor([0.0615, 0.0000, 0.0000, 0.5775, 0.0000, 0.0000, 0.0000, 0.3610, 0.0000])
# the result of this top-three approach are three non-zero probability scores:
