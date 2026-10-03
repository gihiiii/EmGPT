"""
MiniGPT-Em - Training Loop & Pipeline

Trains the MiniGPT model on Eminem lyrics using:
- AdamW with weight decay separation (no decay on 1D weights/biases)
- Cosine learning rate decay with linear warmup
- Gradient clipping
- Periodic validation loss evaluation
- Sample text generation during training
- Checkpoint saving (best and latest)

Usage:
    python training/train.py
    python training/train.py --batch-size 32 --max-iters 3000 --device cuda
"""

import os
import sys
import time
import math
import argparse
from pathlib import Path
import torch
import torch.nn as nn

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from model.transformer import MiniGPT, GPTConfig
from tokenizer.bpe import BPETokenizer
from training.dataset import get_batch, prepare_binary_tokens, load_binary_tokens


# ---------------------------------------------------------------------------
# Training Configurations & Defaults
# ---------------------------------------------------------------------------
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
TOKENIZER_DIR = BASE_DIR / "tokenizer"


def get_lr(step: int, warmup_iters: int, max_iters: int, max_lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with linear warmup."""
    # 1. Linear warmup
    if step < warmup_iters:
        return max_lr * (step + 1) / (warmup_iters + 1)
    # 2. If beyond max_iters, return min_lr
    if step > max_iters:
        return min_lr
    # 3. Cosine decay down to min_lr
    decay_ratio = (step - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def configure_optimizers(model: MiniGPT, weight_decay: float, learning_rate: float, betas: tuple[float, float], device_type: str):
    """
    Split parameters into decayed and non-decayed groups.
    All 2D+ tensors (matmuls, embeddings) are weight decayed.
    All 1D tensors (biases, layernorms) are not decayed.
    Handles weight tying seamlessly without duplicate parameter errors.
    """
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    use_fused = (device_type == "cuda") and ("fused" in torch.optim.AdamW.__init__.__code__.co_varnames)
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    return optimizer


@torch.no_grad()
def estimate_loss(model: MiniGPT, train_data: torch.Tensor, val_data: torch.Tensor, batch_size: int, context_length: int, eval_iters: int, device: str) -> dict[str, float]:
    """Estimate cross-entropy loss over train and validation splits."""
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, batch_size, context_length, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def load_or_cache_tokens(text_path: Path, bin_path: Path, tokenizer: BPETokenizer) -> torch.Tensor:
    """Load pre-tokenized binary file, or tokenize and cache if not present."""
    if bin_path.exists():
        print(f"[Data] Loading cached tokens from {bin_path.name}...")
        try:
            tokens = load_binary_tokens(bin_path)
            print(f"[Data] Loaded {len(tokens):,} tokens from {bin_path.name}")
            return tokens
        except Exception as e:
            print(f"[Data] Failed reading {bin_path.name}: {e}. Re-tokenizing...")

    print(f"[Data] Tokenizing {text_path.name}...")
    text = text_path.read_text(encoding="utf-8")
    token_ids = tokenizer.encode(text)
    print(f"[Data] Encoded {len(token_ids):,} tokens. Saving to {bin_path.name}...")
    prepare_binary_tokens(token_ids, bin_path)
    return torch.tensor(token_ids, dtype=torch.long)


def train():
    parser = argparse.ArgumentParser(description="Train MiniGPT-Em")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--context-length", type=int, default=256, help="Context length / block size (default: 256)")
    parser.add_argument("--max-iters", type=int, default=3000, help="Total training iterations (default: 3000)")
    parser.add_argument("--eval-interval", type=int, default=250, help="Iterations between evaluations (default: 250)")
    parser.add_argument("--eval-iters", type=int, default=40, help="Batches to average for evaluation (default: 40)")
    parser.add_argument("--learning-rate", type=float, default=6e-4, help="Peak learning rate (default: 6e-4)")
    parser.add_argument("--min-lr", type=float, default=6e-5, help="Minimum learning rate (default: 6e-5)")
    parser.add_argument("--warmup-iters", type=int, default=100, help="Warmup iterations (default: 100)")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="AdamW weight decay (default: 0.1)")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient norm clipping (default: 1.0)")
    parser.add_argument("--device", type=str, default="auto", help="Device to use: 'cuda', 'cpu', or 'auto'")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if available")
    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    device_type = "cuda" if "cuda" in device else "cpu"
    print(f"[Training] Using device: {device} ({torch.cuda.get_device_name(0) if device_type == 'cuda' else 'CPU'})")

    # Load Tokenizer
    tok_path = TOKENIZER_DIR / "tokenizer.json"
    if not tok_path.exists():
        print(f"[Training] Tokenizer not found at {tok_path}.")
        print("Please train the tokenizer first with: python tokenizer/bpe.py --train")
        sys.exit(1)

    print("[Training] Loading BPE Tokenizer...")
    tokenizer = BPETokenizer.load(TOKENIZER_DIR)
    actual_vocab_size = len(tokenizer.vocab)
    print(f"[Training] Tokenizer loaded. Vocab size: {actual_vocab_size}")

    # Load and tokenize training and validation datasets
    train_txt = PROCESSED_DIR / "train.txt"
    val_txt = PROCESSED_DIR / "val.txt"
    if not train_txt.exists() or not val_txt.exists():
        print(f"[Training] Processed data not found in {PROCESSED_DIR}.")
        print("Please run `python data/prepare.py` first.")
        sys.exit(1)

    train_bin = PROCESSED_DIR / "train.bin"
    val_bin = PROCESSED_DIR / "val.bin"

    train_data = load_or_cache_tokens(train_txt, train_bin, tokenizer)
    val_data = load_or_cache_tokens(val_txt, val_bin, tokenizer)

    # Initialize Model Config (matching README specification)
    model_config = GPTConfig(
        vocab_size=max(actual_vocab_size, 5000),
        context_length=args.context_length,
        d_model=384,
        n_heads=6,
        n_layers=6,
        d_ff=1536,
        dropout=0.1,
        bias=True,
        tie_weights=True,
    )

    print("=======================================================")
    print("MiniGPT-Em Architecture Specification")
    print("=======================================================")
    print(f"d_model:        {model_config.d_model}")
    print(f"n_heads:        {model_config.n_heads} (head dimension: {model_config.d_model // model_config.n_heads})")
    print(f"n_layers:       {model_config.n_layers}")
    print(f"d_ff:           {model_config.d_ff}")
    print(f"context_length: {model_config.context_length}")
    print(f"vocab_size:     {model_config.vocab_size}")

    model = MiniGPT(model_config)
    total_params = model.num_parameters()
    print(f"Total params:   {total_params:,} (~{total_params / 1e6:.1f}M)")
    print("=" * 55 + "\n")

    model.to(device)

    # Optimizer
    optimizer = configure_optimizers(
        model,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        device_type=device_type,
    )

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    start_iter = 0

    # Resume from checkpoint if requested
    latest_ckpt_path = CHECKPOINTS_DIR / "latest_model.pt"
    if args.resume and latest_ckpt_path.exists():
        print(f"[Training] Resuming from checkpoint: {latest_ckpt_path}")
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_iter = checkpoint.get("iter", 0)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        print(f"[Training] Resumed at step {start_iter} with best val loss {best_val_loss:.4f}")

    # Training Loop
    sample_prompt = "I'm not afraid to"
    print(f"[Training] Starting training for {args.max_iters - start_iter} iterations...")
    t0 = time.time()

    for iter_num in range(start_iter, args.max_iters):
        # Update learning rate
        lr = get_lr(iter_num, args.warmup_iters, args.max_iters, args.learning_rate, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Evaluate loss periodically
        if iter_num % args.eval_interval == 0 or iter_num == args.max_iters - 1:
            losses = estimate_loss(
                model, train_data, val_data, args.batch_size, args.context_length, args.eval_iters, device
            )
            val_loss = losses["val"]
            train_loss = losses["train"]
            dt = time.time() - t0
            print(
                f"step {iter_num:4d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} "
                f"| lr {lr:.2e} | time {dt:.1f}s",
                flush=True,
            )

            # Save checkpoint if best
            if val_loss < best_val_loss or iter_num == args.max_iters - 1:
                best_val_loss = min(val_loss, best_val_loss)
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter": iter_num,
                    "config": model_config,
                    "best_val_loss": best_val_loss,
                }
                save_path = CHECKPOINTS_DIR / "best_model.pt"
                torch.save(ckpt, save_path)
                print(f"  [Checkpoint] Saved best model (val loss {val_loss:.4f}) -> {save_path.name}")

            # Also save latest checkpoint
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter": iter_num,
                    "config": model_config,
                    "best_val_loss": best_val_loss,
                },
                CHECKPOINTS_DIR / "latest_model.pt",
            )

            # Generate quick sample text to preview progress
            model.eval()
            prompt_tokens = tokenizer.encode(sample_prompt)
            x_gen = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
            gen_tokens = model.generate(
                x_gen, max_new_tokens=40, temperature=0.8, top_k=40, repetition_penalty=1.1
            )[0].tolist()
            sample_text = tokenizer.decode(gen_tokens)
            preview = sample_text.replace("\n", " ").strip()
            # Clean non-ASCII for Windows console safety
            safe_preview = preview.encode("ascii", errors="replace").decode("ascii")
            print(f"  [Sample]: \"{safe_preview[:90]}...\"\n", flush=True)
            model.train()
            t0 = time.time()

        # Fetch training batch
        xb, yb = get_batch(train_data, args.batch_size, args.context_length, device)

        # Forward pass
        logits, loss = model(xb, yb)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Gradient clipping
        if args.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # Optimizer step
        optimizer.step()

    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {CHECKPOINTS_DIR / 'best_model.pt'}")
    print("=" * 50)


if __name__ == "__main__":
    train()
