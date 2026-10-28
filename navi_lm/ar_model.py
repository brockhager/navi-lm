"""Decoder-only autoregressive baseline, sized to match the NCM LM1B generator
(12 layers, 12 heads, width 768, RoPE, dropout 0.1, bias, 128-token context)."""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ARConfig:
    vocab_size: int = 30528  # BERT vocab (30522) padded, as in NCM
    real_vocab: int = 30522
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    context_length: int = 128
    dropout: float = 0.1
    bias: bool = True
    rope_base: float = 10000.0


def rope_tables(T: int, head_dim: int, base: float, device):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    freqs = torch.outer(torch.arange(T, device=device).float(), inv)  # (T, hd/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, hd); rotate the two halves of the head dimension
    h = x.size(-1) // 2
    x1, x2 = x[..., :h], x[..., h:]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, c: ARConfig):
        super().__init__()
        self.n_head = c.n_head
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd, bias=c.bias)
        self.proj = nn.Linear(c.n_embd, c.n_embd, bias=c.bias)
        self.p = c.dropout
        self.resid_drop = nn.Dropout(c.dropout)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, C // self.n_head).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.p if self.training else 0.0)
        return self.resid_drop(self.proj(y.transpose(1, 2).reshape(B, T, C)))


class Block(nn.Module):
    def __init__(self, c: ARConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.n_embd)
        self.attn = Attention(c)
        self.ln2 = nn.LayerNorm(c.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(c.n_embd, 4 * c.n_embd, bias=c.bias),
            nn.GELU(),
            nn.Linear(4 * c.n_embd, c.n_embd, bias=c.bias),
            nn.Dropout(c.dropout),
        )

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        return x + self.mlp(self.ln2(x))


class ARLM(nn.Module):
    def __init__(self, c: ARConfig):
        super().__init__()
        self.cfg = c
        self.tok_emb = nn.Embedding(c.vocab_size, c.n_embd)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tied
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("proj.weight") or n.endswith("mlp.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())  # tied weights counted once

    def forward(self, idx, targets=None):
        T = idx.size(1)
        assert T <= self.cfg.context_length
        cos, sin = rope_tables(T, self.cfg.n_embd // self.cfg.n_head, self.cfg.rope_base, idx.device)
        x = self.drop(self.tok_emb(idx))
        for b in self.blocks:
            x = b(x, cos, sin)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, n: int, length: int = 128, temperature: float = 1.0, generator=None):
        """Unconditional ancestral sampling from a BOS token. Returns (n, length) token ids."""
        was_training = self.training
        self.eval()
        dev = next(self.parameters()).device
        idx = torch.zeros(n, 1, dtype=torch.long, device=dev)  # BOS id 0
        for _ in range(length):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logits, _ = self(idx)
            logits = logits[:, -1, : self.cfg.real_vocab].float() / temperature
            nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator)
            idx = torch.cat([idx, nxt], dim=1)
        self.train(was_training)
        return idx[:, 1:]
