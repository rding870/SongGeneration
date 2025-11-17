from datasets import load_dataset
import re
import random
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
import torch
ds = load_dataset("BhavyaMuni/artist-lyrics")

all_songs = []
current_song = []

for artist in ds.keys():
  for item in ds[artist]:
    text = item["text"].strip()
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(rf"\b{artist}\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?mi)^\s*\"?\s*(?:Music|Lyrics)\s*(?:by|:)\s*.*$", "", text)
    text = re.sub(r"(?mi)^\s*(?:Chorus|Verse|Bridge|Prechorus)\s*:?\s*$", "", text)
    text = text.strip()
    if "/n" in text:
      parts = text.split("/n")
      current_song.append(parts[0].strip())
      all_songs.append(current_song)
      current_song = [parts[1].strip()]
    else:
      current_song.append(text)
if current_song:
  all_songs.append(current_song)
print(len(all_songs))
random.shuffle(all_songs)
split_idx = int(0.9 * len(all_songs))
train_songs = all_songs[:split_idx]
val_songs = all_songs[split_idx:]
train_data = "[NL] [endofsong] [NL]".join(["[NL]".join(song) for song in train_songs])
val_data = "[NL] [endofsong] [NL]".join(["[NL]".join(song) for song in val_songs])

print("Example training data:\n", train_data[:10000])


NL = "[NL]"
tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
tok.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.WordPieceTrainer(vocab_size=12000, special_tokens=["[UNK]", "[PAD]", NL], continuing_subword_prefix="##", min_frequency=1)
tok.train_from_iterator([train_data, val_data], trainer)

train_ids = tok.encode(train_data).ids
val_ids   = tok.encode(val_data).ids

train_data = torch.tensor(train_ids, dtype=torch.long)
val_data   = torch.tensor(val_ids, dtype=torch.long)
vocab_size = tok.get_vocab_size()

batch_size   = 64
block_size   = 256
max_iters    = 50_000
eval_interval= 1000
eval_iters   = 200
learning_rate= 1e-3
warmup_steps = 1000
weight_decay = 0.1
grad_clip    = 1.0
n_embd       = 384
n_head       = 6
n_layer      = 8
dropout      = 0.4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)
import torch
import torch.nn as nn
from torch.nn import functional as F

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    assert len(data) > block_size, "block_size is larger than dataset length"
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

class GRUScratch(nn.Module):
  def __init__(self, num_inputs, num_hiddens, sigma=0.01):
    super().__init__()
    self.num_inputs = num_inputs
    self.num_hiddens = num_hiddens
    init_weight = lambda *shape: nn.Parameter(torch.randn(*shape) * sigma)
    triple = lambda: (init_weight(num_inputs, num_hiddens),
                      init_weight(num_hiddens, num_hiddens),
                      nn.Parameter(torch.zeros(num_hiddens)))
    self.W_xz, self.W_hz, self.b_z = triple()
    self.W_xr, self.W_hr, self.b_r = triple()
    self.W_xh, self.W_hh, self.b_h = triple()

  def forward(self, inputs, H=None):
    if H is None:
      H = torch.zeros((inputs.shape[1], self.num_hiddens), device=inputs.device)
    outputs = []
    for X in inputs:
      Z = torch.sigmoid(torch.matmul(X, self.W_xz) +
                        torch.matmul(H, self.W_hz) + self.b_z)
      R = torch.sigmoid(torch.matmul(X, self.W_xr) +
                        torch.matmul(H, self.W_hr) + self.b_r)
      H_tilde = torch.tanh(torch.matmul(X, self.W_xh) +
                           torch.matmul(R * H, self.W_hh) + self.b_h)
      H = Z * H + (1 - Z) * H_tilde
      outputs.append(H)
    outputs = torch.stack(outputs, dim=0)
    return outputs, H

class GRUEncoder(nn.Module):
  def __init__(self, vocab_size, embed_size, num_hiddens):
    super().__init__()
    self.embed = nn.Embedding(vocab_size, embed_size)
    self.core = GRUScratch(embed_size, num_hiddens)

  def forward(self, idx, H=None):
    # idx: [B, T] token ids
    x = self.embed(idx)          # [B, T, E]
    x = x.transpose(0, 1)        # [T, B, E] for GRUScratch
    outputs, H = self.core(x, H) # outputs: [T, B, H]
    return outputs, H            # we only care about H as a representation

class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)
  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x) # (B,T,hs)
    q = self.query(x)
    wei = q@k.transpose(-2, -1) * k.shape[-1]**-0.5
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    v = self.value(x)
    out = wei @ v
    return out
class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)
  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out
class FeedFoward(nn.Module):
  def __init__(self,n_embd):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embd, 4 * n_embd),
        nn.ReLU(),
        nn.Linear(4 * n_embd, n_embd),
        nn.Dropout(dropout),
    )
  def forward(self, x):
    return self.net(x)
@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            X = X.to(device).long()
            Y = Y.to(device).long()
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out
class Block(nn.Module):
  def __init__(self, n_embd, n_head):
    super().__init__()
    head_size = n_embd //n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedFoward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)
  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)
xb, yb = get_batch('train')
n_embd=32
xb = xb.to(device).long()
yb = yb.to(device).long()
class BigramLanguageModel(nn.Module):
  def __init__(self, encoder=None):
    super().__init__()
    self.encoder = encoder                      # NEW

    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(
        Block(n_embd, n_head=4),
        Block(n_embd, n_head=4),
        Block(n_embd, n_head=4),
        nn.LayerNorm(n_embd),
    )
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, idx, targets=None):
    B, T = idx.shape

    # (B, T, C)
    tok_emb = self.token_embedding_table(idx)
    pos = torch.arange(T, device=idx.device)
    pos_emb = self.position_embedding_table(pos).unsqueeze(0)  # (1, T, C)
    x = tok_emb + pos_emb

    # ---- NEW: condition on GRU representation ----
    if self.encoder is not None:
      _, H = self.encoder(idx)           # H: [B, C]  (C == n_embd)
      cond = H.unsqueeze(1)              # [B, 1, C]
      x = x + cond                       # broadcast over time
    # ----------------------------------------------

    x = self.blocks(x)
    logits = self.lm_head(x)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.reshape(B*T, C)
      targets = targets.reshape(B*T)
      loss = F.cross_entropy(logits, targets)
    return logits, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:]
      logits, _ = self(idx_cond)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, 1)
      idx = torch.cat((idx, idx_next), dim=1)
    return idx

m = BigramLanguageModel().to(device)
out, loss = m(xb, yb)
context = torch.zeros((1,1), dtype = torch.long).to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')
    xb = xb.to(device).long()
    yb = yb.to(device).long()
    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)

from tokenizers import decoders
tok.decoder = decoders.WordPiece(prefix="##")
NL = "[NL]"
start_id = tok.token_to_id(NL)
assert start_id is not None, "NL not in vocab"

x = torch.tensor([[start_id]], dtype=torch.long, device=device)
with torch.no_grad():
    y = m.generate(x, max_new_tokens=800)[0].tolist()

out = tok.decode(y[1:], skip_special_tokens=False)

text = out.replace("[NL]", "\n")

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
print(balance_parentheses(text))
