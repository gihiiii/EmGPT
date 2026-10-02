"""
Tests for BPE Tokenizer and Training Dataset (Day 5)
Verifies:
1. Lossless roundtrip encode/decode on sample texts.
2. Handling of special tokens (<|endoftext|>, <|pad|>).
3. PyTorch TextDataset sliding window generation.
4. Input (x) and Target (y) shape and next-token shift: y[t] == x[t+1].
"""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from tokenizer.bpe import BPETokenizer


def test_tokenizer_roundtrip():
    tok = BPETokenizer.load(BASE_DIR / "tokenizer")
    samples = [
        "His palms are sweaty, knees weak, arms are heavy",
        "There's vomit on his sweater already, mom's spaghetti",
        "Snap back to reality, ope there goes gravity",
        "Testing special: <|endoftext|> and <|pad|>",
        "Numbers: 1234567890 and symbols: !@#$%^&*()_+{}[]:;\"'<>,.?/\\|~`",
    ]
    for s in samples:
        tokens = tok.encode(s)
        decoded = tok.decode(tokens)
        assert decoded == s, f"Roundtrip failed for: {s!r}\nGot: {decoded!r}"
    print("[PASS] Tokenizer roundtrip test passed on all test phrases.")


def test_dataset_pipeline():
    import torch
    from training.dataset import TextDataset, get_batch

    tok = BPETokenizer.load(BASE_DIR / "tokenizer")
    text = "Look, if you had one shot, or one opportunity to seize everything you ever wanted..."
    token_ids = tok.encode(text)

    context_length = 8
    dataset = TextDataset(token_ids, context_length=context_length)
    
    assert len(dataset) == max(0, len(token_ids) - context_length), "Dataset length mismatch."

    x, y = dataset[0]
    assert x.shape == torch.Size([context_length]), f"Expected x shape [{context_length}], got {x.shape}"
    assert y.shape == torch.Size([context_length]), f"Expected y shape [{context_length}], got {y.shape}"
    
    # Verify next-token prediction shift: y must be x shifted by 1 token
    for t in range(context_length - 1):
        assert x[t + 1] == y[t], f"Target shift mismatch at position {t}: x[t+1]={x[t+1]}, y[t]={y[t]}"
    
    # Test batching utility
    data_tensor = torch.tensor(token_ids, dtype=torch.long)
    batch_size = 4
    batch_x, batch_y = get_batch(data_tensor, batch_size=batch_size, context_length=context_length, device="cpu")
    assert batch_x.shape == torch.Size([batch_size, context_length])
    assert batch_y.shape == torch.Size([batch_size, context_length])
    print("[PASS] TextDataset shapes, slice shifts, and batching verified successfully.")


if __name__ == "__main__":
    test_tokenizer_roundtrip()
    try:
        test_dataset_pipeline()
    except ImportError:
        print("[SKIP] PyTorch is still installing, dataset test will run after install completes.")
