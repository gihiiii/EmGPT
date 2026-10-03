"""
MiniGPT-Em - Transformer Architecture

Pure PyTorch implementation of a causal decoder-only Transformer language model.
Implements multi-head causal self-attention, feed-forward network with GELU,
pre-layer normalization, and residual connections.

Model Specification:
    - d_model: 384
    - n_heads: 6 (d_head = 64)
    - n_layers: 6
    - d_ff: 1536
    - context_length: 256
    - vocab_size: ~5,000
    - Total parameters: ~10.6M (with weight tying)
"""

import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 5000
    context_length: int = 256
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 6
    d_ff: int = 1536
    dropout: float = 0.1
    bias: bool = True
    tie_weights: bool = True


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.d_head = config.d_model // config.n_heads
        self.dropout = config.dropout

        # Combined Q, K, V projection
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        # Dropouts
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask buffer
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.context_length, config.context_length)).view(
                1, 1, config.context_length, config.context_length
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (d_model)

        # Calculate query, key, values
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Split heads: (B, T, C) -> (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Check if PyTorch's native scaled_dot_product_attention is available
        if hasattr(F, "scaled_dot_product_attention"):
            # Native FlashAttention / memory-efficient causal attention
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Fallback manual causal attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        # Concatenate heads: (B, nh, T, hs) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection with residual dropout
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Position-wise Feed-Forward Network."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.act = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Standard Transformer Decoder Block with Pre-LayerNorm."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN residual connections
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MiniGPT(nn.Module):
    """
    MiniGPT: Decoder-only Autoregressive Transformer Language Model.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.d_model),
                wpe=nn.Embedding(config.context_length, config.d_model),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)]),
                ln_f=nn.LayerNorm(config.d_model),
            )
        )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying between token embeddings and output projection
        if config.tie_weights:
            self.transformer.wte.weight = self.lm_head.weight

        # Initialize all weights
        self.apply(self._init_weights)

        # Apply special scaled init to residual projections (per GPT-2 paper)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def num_parameters(self, non_embedding: bool = False) -> int:
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.context_length
        ), f"Cannot forward sequence of length {t}, context length is {self.config.context_length}"

        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # Forward through embeddings
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, d_model)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, d_model)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Forward through transformer blocks
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            # If we are given targets, compute cross entropy loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference optimization: forward only the last token position
            logits = self.lm_head(x[:, [-1], :])  # shape (b, 1, vocab_size)
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with temperature, top-k, top-p (nucleus),
        and repetition penalty.

        Args:
            idx: (B, T) tensor of context token IDs
            max_new_tokens: maximum number of tokens to generate
            temperature: 1.0 = default, <1.0 = more deterministic, >1.0 = more creative
            top_k: filter to top K most likely tokens (None or <= 0 disables)
            top_p: nucleus sampling cutoff (None or >= 1.0 disables)
            repetition_penalty: > 1.0 penalizes tokens that have already appeared
            eos_token_id: token ID that terminates generation
        """
        for _ in range(max_new_tokens):
            # Crop context if it exceeds context_length
            idx_cond = (
                idx
                if idx.size(1) <= self.config.context_length
                else idx[:, -self.config.context_length :]
            )

            # Forward model to get logits for the next token
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # (B, vocab_size)

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for b in range(idx.size(0)):
                    unique_tokens = torch.unique(idx[b])
                    for t in unique_tokens:
                        if logits[b, t] < 0:
                            logits[b, t] *= repetition_penalty
                        else:
                            logits[b, t] /= repetition_penalty

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature
            else:
                # Greedy decoding (argmax)
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                idx = torch.cat((idx, next_token), dim=1)
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break
                continue

            # Optional Top-K filtering
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # Optional Top-P (nucleus) filtering
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right to keep at least the first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                for b in range(logits.size(0)):
                    indices_to_remove = sorted_indices[b, sorted_indices_to_remove[b]]
                    logits[b, indices_to_remove] = -float("Inf")

            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            idx = torch.cat((idx, next_token), dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return idx
