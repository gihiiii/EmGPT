"""
MiniGPT-Em - Text Generation & Sampling Script

Generates lyrics autoregressively using a trained MiniGPT checkpoint.
Supports temperature scaling, top-k filtering, nucleus (top-p) sampling,
and repetition penalty.

Usage:
    python generate.py
    python generate.py --prompt "I'm not afraid to" --temperature 0.85 --top-k 40
    python generate.py --prompt "Look, if you had" --max-new-tokens 150 --num-samples 3
"""

import sys
import argparse
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from model.transformer import MiniGPT, GPTConfig
from tokenizer.bpe import BPETokenizer


def load_model(checkpoint_path: Path, device: str) -> tuple[MiniGPT, GPTConfig]:
    """Load MiniGPT weights and config from checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = MiniGPT(config)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model, config


def generate(
    model: MiniGPT,
    tokenizer: BPETokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    device: str = "cpu",
) -> str:
    """Generate text continuation from prompt."""
    prompt_tokens = tokenizer.encode(prompt)
    if not prompt_tokens:
        prompt_tokens = [tokenizer.eot_token]

    x = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        out_tokens = model.generate(
            x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=tokenizer.eot_token,
        )[0].tolist()

    return tokenizer.decode(out_tokens)


def main():
    parser = argparse.ArgumentParser(description="Generate lyrics using MiniGPT-Em")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(BASE_DIR / "checkpoints" / "best_model.pt"),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=str,
        default=str(BASE_DIR / "tokenizer"),
        help="Path to tokenizer directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Look, if you had one shot",
        help="Prompt to prime generation",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=120,
        help="Number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (lower = more focused, higher = more creative)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Top-K sampling cutoff (0 to disable)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-P nucleus sampling threshold (1.0 to disable)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.15,
        help="Penalty for repeating tokens (>1.0 reduces loops)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of independent completions to generate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device ('cuda', 'cpu', or 'auto')",
    )
    args = parser.parse_args()

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Loading checkpoint from: {args.checkpoint}")
    model, config = load_model(Path(args.checkpoint), device=device)
    tokenizer = BPETokenizer.load(args.tokenizer_dir)

    print("=" * 60)
    print(f"MiniGPT-Em Generator (~{model.num_parameters() / 1e6:.1f}M params)")
    print(f"Prompt: {repr(args.prompt)}")
    print(f"Params: temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}, rep_pen={args.repetition_penalty}")
    print("=" * 60 + "\n")

    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"--- Sample {i + 1} of {args.num_samples} ---")
        output_text = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            top_p=args.top_p if args.top_p < 1.0 else None,
            repetition_penalty=args.repetition_penalty,
            device=device,
        )
        safe_output = output_text.encode("ascii", errors="replace").decode("ascii")
        print(safe_output)
        print("\n" + "-" * 40 + "\n")


if __name__ == "__main__":
    main()
