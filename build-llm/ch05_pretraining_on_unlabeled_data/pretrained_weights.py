import tiktoken
import torch
import numpy as np
from config import GPT_CONFIG_124M
from gpt_download import load_downloaded_gpt2
from ch04_code import GPTModel
from main import generate, text_to_token_ids, token_ids_to_text

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

model_name = "gpt2-small (124M)"
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024})


settings, params = load_downloaded_gpt2(model_size="124M", models_dir="gpt2")
# Settings: {'n_vocab': 50257, 'n_ctx': 1024, 'n_embd': 768, 'n_head': 12, 'n_layer': 12}
# Parameter dictionary keys: dict_keys(['blocks', 'b', 'g', 'wpe', 'wte'])
#
print(params["wte"])
print("Token embedding weight tensor dimensions:", params["wte"].shape)
# Token embedding weight tensor dimensions: (50257, 768)


# OpenAI used bias vectors in the multi-head attention module’s linear layers to implement the query,
# key, and value matrix computations. Bias vectors are not commonly used in LLMs anymore as they don’t
# improve the modeling performance and are thus unnecessary. However, since we are working with
# pretrained weights, we need to match the settings for consistency and enable these bias vectors:
NEW_CONFIG.update({"qkv_bias": True})

gpt = GPTModel(NEW_CONFIG)
gpt.eval()


# By default, the GPTModel instance is initialized with random weights for pretraining. The last step
# to using OpenAI’s model weights is to override these random weights with the weights we loaded into
# the params dictionary. For this, we will first define a small assign utility function that checks
# whether two tensors or arrays (left and right ) have the same dimensions or shape and returns the
# right tensor as trainable PyTorch parameters:
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")

    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    # Iterates over each transformer block in the model
    for b in range(len(params["blocks"])):
        # The np.split function is used to divide the attention and bias weights
        # into three equal parts for the query, key, and value components.
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1
        )
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T
        )
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T
        )
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T
        )

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1
        )
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b
        )
        gpt.trf_blocks[b].att.W_key.bias = assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b
        )

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"],
        )

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T,
        )
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"]
        )
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"],
        )

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"]
        )
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"]
        )
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"]
        )
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"]
        )

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


# In the load_weights_into_gpt function, we carefully match the weights from OpenAI’s implementation
# with our GPTModel implementation. To pick a specific example, OpenAI stored the weight tensor for
# the output projection layer for the first transformer block as params["blocks"][0]["attn"]["c_proj"]["w"].
# In our implementation, this weight tensor corresponds to gpt.trf_blocks[b].att.out_proj .weight, where
# gpt is a GPTModel instance. Developing the load_weights_into_gpt function took a lot of guesswork since
# OpenAI used a slightly different naming convention from ours. However, the assign function would alert
# us if we try to match two tensors with different dimensions. Also, if we made a mistake in this
# function, we would notice this, as the resulting GPT model would be unable to produce coherent text.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_weights_into_gpt(gpt, params)
gpt.to(device)


tokenizer = tiktoken.get_encoding("gpt2")

torch.manual_seed(123)
token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5,
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
