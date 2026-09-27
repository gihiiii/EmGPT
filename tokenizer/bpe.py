"""
MiniGPT-Em - Byte-Pair Encoding (BPE) Tokenizer from Scratch

A byte-level BPE tokenizer inspired by GPT-2, built without external NLP libraries.
Because it operates at the byte level, every UTF-8 string can be represented
without any out-of-vocabulary (OOV) errors.

Usage:
    # Train tokenizer on processed text
    python tokenizer/bpe.py --train --vocab-size 5000
    # Test encoding / decoding
    python tokenizer/bpe.py --test "I'm beginnin' to feel like a Rap God, Rap God"
"""

import os
import re
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "processed" / "train.txt"
DEFAULT_SAVE_DIR = BASE_DIR / "tokenizer"


# GPT-2 style pre-tokenization regex
# Splits on contractions, letters, numbers, punctuation, and whitespace
GPT2_SPLIT_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"""
)


class BPETokenizer:
    """
    Byte-level BPE Tokenizer from scratch.

    Base vocab:
        0..255: Single byte values
        256: <|endoftext|>
        257: <|pad|>
    Followed by learned merged tokens up to `vocab_size`.
    """

    SPECIAL_TOKENS = {
        "<|endoftext|>": 256,
        "<|pad|>": 257,
    }

    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.special_tokens = dict(self.SPECIAL_TOKENS)
        self.special_tokens_inv = {v: k for k, v in self.special_tokens.items()}

        # merges: tuple(token_id_1, token_id_2) -> merged_token_id
        self.merges: dict[tuple[int, int], int] = {}
        # vocab: token_id -> bytes
        self.vocab: dict[int, bytes] = {}
        # inverse vocab: bytes -> token_id
        self.inv_vocab: dict[bytes, int] = {}

        self._init_base_vocab()

    def _init_base_vocab(self):
        """Initialize the base 256 byte vocabulary and special tokens."""
        self.vocab = {i: bytes([i]) for i in range(256)}
        for token_str, token_id in self.special_tokens.items():
            self.vocab[token_id] = token_str.encode("utf-8")
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    @property
    def eot_token(self) -> int:
        return self.special_tokens["<|endoftext|>"]

    @property
    def pad_token(self) -> int:
        return self.special_tokens["<|pad|>"]

    def _get_stats(self, words_and_counts: list[tuple[list[int], int]]) -> dict[tuple[int, int], int]:
        """Count frequency of all adjacent token pairs."""
        stats = defaultdict(int)
        for tokens, count in words_and_counts:
            for i in range(len(tokens) - 1):
                stats[(tokens[i], tokens[i + 1])] += count
        return stats

    def _merge_tokens(self, tokens: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """Replace all occurrences of `pair` in `tokens` with `new_id`."""
        new_tokens = []
        i = 0
        p0, p1 = pair
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == p0 and tokens[i + 1] == p1:
                new_tokens.append(new_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def train(self, text: str, verbose: bool = True):
        """Train BPE merges on raw text until vocab_size is reached."""
        self._init_base_vocab()
        self.merges = {}

        # Number of merges needed
        num_special = len(self.special_tokens)
        base_count = 256 + num_special
        num_merges = self.vocab_size - base_count

        if num_merges <= 0:
            if verbose:
                print(f"[Tokenizer] Vocab size {self.vocab_size} <= base count {base_count}, no merges needed.")
            return

        if verbose:
            print(f"[Tokenizer] Training BPE: Target vocab {self.vocab_size}, learning {num_merges} merges...")

        # 1. Pre-tokenize text into words using regex
        # Special tokens are excluded from splitting
        parts = re.split(r"(<\|endoftext\|>|<\|pad\|>)", text)
        words_list = []
        for part in parts:
            if part in self.special_tokens:
                continue
            words_list.extend(GPT2_SPLIT_PATTERN.findall(part))

        word_counts = Counter(words_list)
        # Convert each word into a list of byte IDs
        words_and_counts = [
            (list(word.encode("utf-8")), count) for word, count in word_counts.items() if word
        ]

        # 2. Iteratively merge the most frequent pair
        next_id = base_count
        log_interval = max(1, num_merges // 10)

        for step in range(num_merges):
            stats = self._get_stats(words_and_counts)
            if not stats:
                if verbose:
                    print(f"[Tokenizer] No more pairs to merge at step {step + 1}.")
                break

            # Find pair with highest frequency
            best_pair = max(stats, key=stats.get)
            if stats[best_pair] < 2:
                # Pair occurred less than twice, stop early
                if verbose:
                    print(f"[Tokenizer] Stopping: max pair frequency is {stats[best_pair]} at step {step + 1}.")
                break

            new_token_id = next_id
            self.merges[best_pair] = new_token_id
            self.vocab[new_token_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.inv_vocab[self.vocab[new_token_id]] = new_token_id

            # Apply merge to all words
            words_and_counts = [
                (self._merge_tokens(tokens, best_pair, new_token_id), count)
                for tokens, count in words_and_counts
            ]

            next_id += 1

            if verbose and ((step + 1) % log_interval == 0 or step == num_merges - 1):
                pair_repr = (
                    self.vocab[best_pair[0]].decode("utf-8", errors="replace")
                    + " + "
                    + self.vocab[best_pair[1]].decode("utf-8", errors="replace")
                )
                print(
                    f"  [Merge {step + 1:4d}/{num_merges}] ({best_pair[0]}, {best_pair[1]}) -> {new_token_id} "
                    f"freq: {stats[best_pair]:6d} | repr: {repr(pair_repr)}"
                )

        if verbose:
            print(f"[Tokenizer] Training complete. Final vocab size: {len(self.vocab)}")

    def _encode_chunk(self, chunk_bytes: bytes) -> list[int]:
        """Encode a single pre-tokenized chunk using learned merges."""
        tokens = list(chunk_bytes)
        if len(tokens) <= 1:
            return tokens

        while len(tokens) >= 2:
            # Find the best merge pair in the current token list
            # We select the pair that appeared earliest in self.merges (lowest merge rank)
            best_pair = None
            min_rank = float("inf")
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merges.get(pair)
                if rank is not None and rank < min_rank:
                    min_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            tokens = self._merge_tokens(tokens, best_pair, self.merges[best_pair])

        return tokens

    def encode(self, text: str, allowed_special: bool = True) -> list[int]:
        """
        Encode string text into a list of token IDs.
        Handles special tokens like <|endoftext|> and <|pad|>.
        """
        if not text:
            return []

        # Split on special tokens if allowed
        if allowed_special:
            pattern = "(" + "|".join(re.escape(k) for k in self.special_tokens.keys()) + ")"
            parts = re.split(pattern, text)
        else:
            parts = [text]

        all_tokens = []
        for part in parts:
            if not part:
                continue
            if allowed_special and part in self.special_tokens:
                all_tokens.append(self.special_tokens[part])
            else:
                chunks = GPT2_SPLIT_PATTERN.findall(part)
                for chunk in chunks:
                    chunk_bytes = chunk.encode("utf-8")
                    all_tokens.extend(self._encode_chunk(chunk_bytes))

        return all_tokens

    def decode(self, tokens: list[int]) -> str:
        """Decode a list of token IDs back into a Unicode string."""
        byte_parts = []
        for t in tokens:
            if t in self.special_tokens_inv:
                # Represent special tokens directly as text
                byte_parts.append(self.special_tokens_inv[t].encode("utf-8"))
            elif t in self.vocab:
                byte_parts.append(self.vocab[t])
            else:
                byte_parts.append(f"<|unk_{t}|>".encode("utf-8"))

        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def save(self, directory: Path | str):
        """Save the tokenizer merges and configuration to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        config = {
            "vocab_size": self.vocab_size,
            "actual_vocab_size": len(self.vocab),
            "special_tokens": self.special_tokens,
            "merges": [
                [p[0], p[1], new_id] for p, new_id in self.merges.items()
            ],
        }

        save_path = directory / "tokenizer.json"
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        print(f"[Tokenizer] Saved model to {save_path}")

    @classmethod
    def load(cls, directory: Path | str) -> "BPETokenizer":
        """Load a trained BPE tokenizer from disk."""
        directory = Path(directory)
        load_path = directory / "tokenizer.json"

        if not load_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {load_path}")

        with open(load_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        tok = cls(vocab_size=config["vocab_size"])
        tok.special_tokens = config.get("special_tokens", dict(cls.SPECIAL_TOKENS))
        tok.special_tokens_inv = {v: k for k, v in tok.special_tokens.items()}
        tok._init_base_vocab()

        for p0, p1, new_id in config["merges"]:
            pair = (p0, p1)
            tok.merges[pair] = new_id
            tok.vocab[new_id] = tok.vocab[p0] + tok.vocab[p1]
            tok.inv_vocab[tok.vocab[new_id]] = new_id

        return tok


def main():
    parser = argparse.ArgumentParser(description="Train and test MiniGPT-Em BPE Tokenizer")
    parser.add_argument("--train", action="store_true", help="Train tokenizer on data/processed/train.txt")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH), help="Path to training corpus")
    parser.add_argument("--vocab-size", type=int, default=5000, help="Target vocabulary size (default: 5000)")
    parser.add_argument("--save-dir", type=str, default=str(DEFAULT_SAVE_DIR), help="Directory to save tokenizer")
    parser.add_argument("--test", type=str, default=None, help="Test string to encode and decode")
    args = parser.parse_args()

    if args.train:
        data_path = Path(args.data)
        if not data_path.exists():
            print(f"[Tokenizer] Error: Training file not found at {data_path}")
            print("Please run `python data/prepare.py` first.")
            return

        print(f"[Tokenizer] Reading {data_path}...")
        text = data_path.read_text(encoding="utf-8")
        print(f"[Tokenizer] Loaded {len(text):,} characters.")

        tokenizer = BPETokenizer(vocab_size=args.vocab_size)
        tokenizer.train(text, verbose=True)
        tokenizer.save(args.save_dir)

        # Quick self-test
        sample = "Look, if you had one shot, or one opportunity\nTo seize everything you ever wanted..."
        tokens = tokenizer.encode(sample)
        decoded = tokenizer.decode(tokens)
        print("\n--- Self-Test ---")
        print(f"Original: {repr(sample)}")
        print(f"Tokens:   {tokens[:20]}... ({len(tokens)} total)")
        print(f"Decoded:  {repr(decoded)}")
        print(f"Lossless: {sample == decoded}")

    elif args.test:
        save_path = Path(args.save_dir) / "tokenizer.json"
        if not save_path.exists():
            print(f"Tokenizer not found at {save_path}. Run with --train first.")
            return

        tokenizer = BPETokenizer.load(args.save_dir)
        tokens = tokenizer.encode(args.test)
        decoded = tokenizer.decode(tokens)
        print("Test Text:", args.test)
        print(f"Token IDs ({len(tokens)} tokens):", tokens)
        print("Reconstructed:", decoded)
        print("Lossless:", args.test == decoded)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
