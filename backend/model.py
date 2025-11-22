import torch
import torch.nn as nn
from torch.nn import functional as F

# ============================================================
#                     GRU ENCODER
# ============================================================

class GRUScratch(nn.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        init_weight = lambda *shape: nn.Parameter(torch.randn(*shape) * sigma)
        triple = lambda: (
            init_weight(num_inputs, num_hiddens),
            init_weight(num_hiddens, num_hiddens),
            nn.Parameter(torch.zeros(num_hiddens)),
        )
        self.W_xz, self.W_hz, self.b_z = triple()
        self.W_xr, self.W_hr, self.b_r = triple()
        self.W_xh, self.W_hh, self.b_h = triple()

    def forward(self, inputs, H=None):
        # inputs: [T, B, E]
        T, B, Hdim = inputs.shape
        if H is None:
            H = torch.zeros((B, Hdim), device=inputs.device)

        outputs = []
        for X in inputs:
            Z = torch.sigmoid(X @ self.W_xz + H @ self.W_hz + self.b_z)
            R = torch.sigmoid(X @ self.W_xr + H @ self.W_hr + self.b_r)
            H_tilde = torch.tanh(X @ self.W_xh + (R * H) @ self.W_hh + self.b_h)
            H = Z * H + (1 - Z) * H_tilde
            outputs.append(H)

        return torch.stack(outputs, dim=0), H


class GRUEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.core = GRUScratch(embed_size, num_hiddens)

    def forward(self, idx, H=None):
        # idx: [B, T]
        x = self.embed(idx).transpose(0, 1)  # -> [T, B, E]
        return self.core(x, H)


# ============================================================
#                   TRANSFORMER BLOCKS
# ============================================================

class Head(nn.Module):
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)        # (B, T, hs)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedFoward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# ============================================================
#              BIGRAM LANGUAGE MODEL  (MATCHES TRAINING)
# ============================================================

class BigramLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_embd=32,
        n_head=4,
        n_layer=3,
        block_size=256,
        dropout=0.4,
        encoder=None,
    ):
        super().__init__()
        self.encoder = encoder

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(
            *[
                Block(n_embd, n_head, block_size, dropout)
                for _ in range(n_layer)
            ],
            nn.LayerNorm(n_embd),
        )

        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)
        ).unsqueeze(0)
        x = tok_emb + pos_emb

        if self.encoder is not None:
            _, H = self.encoder(idx)   # H: [B, n_embd]
            x = x + H.unsqueeze(1)

        x = self.blocks(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        loss = F.cross_entropy(
            logits.view(B * T, -1),
            targets.view(B * T),
        )
        return logits, loss

    def generate(self, idx, max_new_tokens, end_id=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, 1)
            idx = torch.cat((idx, idx_next), dim=1)
            if end_id is not None and idx_next.item() == end_id:
                break
        return idx
