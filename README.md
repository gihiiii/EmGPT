# 🎤 EmGPT: ~12.7M Parameter Language Model from Scratch

A decoder-only autoregressive GPT language model built **from scratch in pure PyTorch** (no Hugging Face transformers, no external tokenization libraries), trained on Eminem's discography.

Feed it a prompt and watch it spit bars:

```
Prompt: "Look, if you had one shot"
Output:
Look, if you had one shot up in a little bit of all get fucked up
And now, but I'ma be alright with you're gone

[Verse 3]
I can't want the door and swip into your block
But it, it's like I'm the brinkin'
It feels like to go, here I'm sickness...
```

---

## 🏗️ Architecture & Specifications

Implemented from first principles in `model/transformer.py`:

```
Input Tokens
     │
     ▼
[Token Embeddings (wte)] + [Learned Positional Embeddings (wpe)]
     │
     ▼
[Transformer Block × 6]
  ├─ LayerNorm ─► Multi-Head Causal Self-Attention (6 heads) ─► Dropout ─► (+) Residual
  └─ LayerNorm ─► Feed-Forward MLP (GELU, 4x expansion)      ─► Dropout ─► (+) Residual
     │
     ▼
Final LayerNorm (ln_f)
     │
     ▼
Linear Language Model Head (lm_head) [Weight-Tied to wte]
     │
     ▼
Logits (Softmax + Top-K / Top-P / Temperature / Repetition Penalty)
```

| Hyperparameter | Value | Description |
|---|---|---|
| `d_model` | 384 | Embedding dimensionality |
| `n_heads` | 6 | Multi-head attention heads (`d_head = 64`) |
| `n_layers` | 6 | Stacked transformer decoder blocks |
| `d_ff` | 1536 | Feed-forward inner expansion (`4 × d_model`) |
| `context_length` | 256 | Maximum sequence attention window |
| `vocab_size` | 5,000 | Learned Byte-Pair Encoding (BPE) vocabulary |
| `tie_weights` | True | Memory & parameter sharing (`wte.weight == lm_head.weight`) |
| **Total Parameters** | **12,665,856** | **~12.7M parameters** |

---

## 📈 Training Dynamics & Results

- **Dataset:** 99 songs (~486,000 characters / ~132,000 tokens)
- **Optimizer:** AdamW with weight decay separation (decay on 2D weights, 0.0 on biases & layernorms)
- **Schedule:** Linear warmup (100 steps) + Cosine decay down to 10% peak learning rate
- **Gradient Clipping:** Max norm of `1.0`

| Iteration | Train Loss | Val Loss | Learning Rate | Sample Preview |
|---|---|---|---|---|
| **Step 0** | 8.5723 | 8.5699 | $5.94 \times 10^{-6}$ | Random uncompressed byte noise |
| **Step 100** | 5.0537 | 5.4094 | $6.00 \times 10^{-4}$ | Basic words, quotes, punctuation |
| **Step 400** | 3.8651 | **5.0503 (Best)** | $4.65 \times 10^{-4}$ | Verse structure tags (`[Chorus]`, `[Verse 3]`) |
| **Step 700** | 2.9722 | 5.0796 | $1.95 \times 10^{-4}$ | Rhyme cadences, Eminem references |
| **Step 1000** | **2.5279** | 5.2362 | $6.00 \times 10^{-5}$ | Complete multi-line song structure |

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/gihiiii/EmGPT.git
cd EmGPT

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Generate from Terminal
```bash
python generate.py --prompt "I'm not afraid to" --temperature 0.8 --top-k 40
```

### 3. Launch Interactive Web UI
```bash
python app.py
```
Open [http://localhost:7860](http://localhost:7860) in your browser to interactively adjust temperature, top-k, top-p, and repetition penalty sliders with real-time output generation.

---

## 🧪 Testing

Run the automated test suite verifying tokenizer roundtrips, context slicing, and next-token target alignment ($y_t = x_{t+1}$):
```bash
python tests/test_tokenizer_and_dataset.py
```

---

## 📚 Study Guide / Interview Prep

An Anki-compatible flashcard deck covering core LLM concepts (BPE tokenization, attention mathematics, training dynamics, sampling strategies, and base vs. instruction-tuned models) is included:
- Importable deck: [`docs/anki_flashcards.txt`](docs/anki_flashcards.txt)

---

## 📜 License
MIT
