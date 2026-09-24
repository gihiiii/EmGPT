 EmGPT

A ~10M parameter GPT language model built **from scratch** in pure PyTorch, trained on Eminem lyrics.

Feed it a prompt like *"I'm not afraid to..."* and watch it spit bars.

## Why?

To deeply understand how large language models work — not by using a library, but by building every component from the ground up:

- **BPE Tokenizer** — byte-pair encoding, implemented from scratch
- **Transformer Architecture** — multi-head causal self-attention, feed-forward networks, residual connections, layer norm
- **Training Pipeline** — AdamW, cosine LR schedule, gradient clipping, checkpointing
- **Text Generation** — temperature, top-k, top-p (nucleus) sampling, repetition penalty

## Model Architecture

```
Input → BPE Tokenizer → Token Embeddings + Positional Embeddings
  → [Transformer Block × 6] → Layer Norm → Linear → Logits
```

| Hyperparameter | Value |
|---|---|
| `d_model` | 384 |
| `n_heads` | 6 |
| `n_layers` | 6 |
| `d_ff` | 1536 |
| `context_length` | 256 |
| `vocab_size` | ~5,000 |
| **Total params** | **~10.6M** |

## Project Structure

```
minigpt-em/
├── data/               # Lyrics data pipeline
├── tokenizer/          # BPE tokenizer from scratch
├── model/              # Transformer architecture
├── training/           # Training loop & utilities
├── generate.py         # Text generation
├── app.py              # Gradio demo UI
└── notebooks/          # Exploration & experiments
```

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/minigpt-em.git
cd minigpt-em

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

## Training

```bash
# 1. Prepare the data
python data/prepare.py

# 2. Train the tokenizer
python tokenizer/bpe.py

# 3. Train the model
python training/train.py
```

## Generate Text

```bash
python generate.py --prompt "I'm not afraid to" --temperature 0.8 --top_k 40
```

## License

MIT
