"""
MiniGPT-Em - Dataset & Data Loading Utilities

Provides dataset handling and high-throughput batch generation for training
autoregressive language models.
"""

from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np


class TextDataset(Dataset):
    """
    Standard PyTorch Dataset for sequential context_length chunks.
    """

    def __init__(self, token_ids: list[int] | torch.Tensor, context_length: int = 256):
        if isinstance(token_ids, list):
            self.data = torch.tensor(token_ids, dtype=torch.long)
        else:
            self.data = token_ids.to(dtype=torch.long)

        self.context_length = context_length

    def __len__(self) -> int:
        return max(0, len(self.data) - self.context_length)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.context_length + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate a random small batch of inputs x and targets y directly on device.
    Fast tensor indexing avoiding DataLoader overhead.
    """
    max_idx = len(data) - context_length - 1
    if max_idx <= 0:
        raise ValueError(
            f"Data length ({len(data)}) must be greater than context_length + 1 ({context_length + 1})"
        )

    ix = torch.randint(0, max_idx, (batch_size,))
    x = torch.stack([data[i : i + context_length] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + context_length] for i in ix])

    if device != "cpu":
        # Pin memory transfer if on CPU initially
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    return x, y


def prepare_binary_tokens(tokens: list[int], output_path: Path):
    """Save token IDs as uint16 binary file for ultra-fast loading."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(tokens, dtype=np.uint16)
    arr.tofile(output_path)


def load_binary_tokens(file_path: Path) -> torch.Tensor:
    """Load token IDs from uint16 binary file into a 1D torch Tensor."""
    arr = np.fromfile(file_path, dtype=np.uint16)
    return torch.from_numpy(arr.astype(np.int64))
