from flask import Flask, request, jsonify, send_from_directory
import torch
from tokenizers import Tokenizer
import os

# -------- MODEL IMPORT -------- #
from model import (
    BigramLanguageModel,
    GRUEncoder
)

app = Flask(__name__)

# -------- CONSTANTS -------- #
MODEL_PATH = "lyrics_model_fri.pt"
TOKENIZER_PATH = "tokenizer.json"

device = torch.device("cpu")

# -------- Load Tokenizer -------- #
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
vocab_size = tokenizer.get_vocab_size()

# -------- Rebuild EXACT SAME architecture used in training -------- #
n_embd = 32          # MUST MATCH TRAINING
n_head = 4           # MUST MATCH TRAINING
n_layer = 3          # MUST MATCH TRAINING
block_size = 256     # MUST MATCH TRAINING
dropout = 0.4

# Build encoder EXACTLY as before
encoder = GRUEncoder(
    vocab_size=vocab_size,
    embed_size=n_embd,
    num_hiddens=n_embd
).to(device)

# Build language model EXACTLY as before
model = BigramLanguageModel(
    vocab_size=vocab_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    block_size=block_size,
    dropout=dropout,
    encoder=encoder
).to(device)

# -------- Load Weights -------- #
state = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state, strict=True)
model.eval()

def balance_parentheses(line: str) -> str:
    stack = 0
    output = []

    for ch in line:
        if ch == '(':
            stack += 1
            output.append(ch)
        elif ch == ')':
            if stack > 0:
                stack -= 1
                output.append(ch)
        else:
            output.append(ch)

    output.extend(')' * stack)

    return ''.join(output)

# -------- Helper: tokenize -------- #
def tokenize_input(text):
    ids = tokenizer.encode(text).ids
    return torch.tensor([ids], dtype=torch.long, device=device)

@app.route("/")
def serve_react():
    return send_from_directory(app.static_folder, "index.html")

# -------- API ENDPOINT -------- #
@app.route("/generate", methods=["POST"])
def generate():
    x = tokenize_input("[NL]")
    stop_condition = tokenizer.token_to_id("[endofsong]")

    with torch.no_grad():
        y = model.generate(x, max_new_tokens=1000, end_id=stop_condition)[0].tolist()

    text = tokenizer.decode(y, skip_special_tokens=False)
    text = text.replace("[NL]", "\n")
    text = text.replace("[endofsong]", "")
    text = balance_parentheses(text)
    return jsonify({"text": text})

# -------- Flask Run -------- #
if __name__ == "__main__":
    app.run(port=5000, debug=True)
