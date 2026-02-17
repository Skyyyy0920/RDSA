"""Dataset classes for RDSA training and evaluation.

Provides:
- ContrastDataset: Safe/unsafe prompt pairs for subspace identification
- RDSATrainDataset: Mixed harmful/benign samples for RDSA fine-tuning
- SafetyEvalDataset: AdvBench harmful instructions for attack evaluation

CRITICAL: Reproducibility — all shuffling uses explicit seeds.
"""

from __future__ import annotations

import csv
import json
import random
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


class ContrastDataset(Dataset):
    """Dataset of safe/unsafe prompt pairs for subspace identification.

    Loads from JSONL files where each line has the format:
    ``{"prompt": "...", "image_path": "..."}``  (image_path is optional)

    Pairs are constructed by matching indices: ``safe[i]`` pairs with
    ``unsafe[i]``. The number of pairs is ``min(len(safe), len(unsafe))``.

    Args:
        safe_path: Path to safe prompts JSONL file.
        unsafe_path: Path to unsafe prompts JSONL file.
        processor: HuggingFace processor for tokenization.
        max_length: Maximum sequence length after tokenization.
    """

    def __init__(
        self,
        safe_path: str,
        unsafe_path: str,
        processor: Any,
        max_length: int = 512,
    ) -> None:
        self.processor = processor
        self.max_length = max_length

        self.safe_samples = self._load_jsonl(safe_path)
        self.unsafe_samples = self._load_jsonl(unsafe_path)

        self._length = min(len(self.safe_samples), len(self.unsafe_samples))

    @staticmethod
    def _load_jsonl(path: str) -> list[dict[str, str]]:
        """Load samples from a JSONL file."""
        samples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        safe = self.safe_samples[idx]
        unsafe = self.unsafe_samples[idx]

        safe_enc = self._encode(safe["prompt"])
        unsafe_enc = self._encode(unsafe["prompt"])

        return {
            "safe_input_ids": safe_enc["input_ids"],
            "safe_attention_mask": safe_enc["attention_mask"],
            "unsafe_input_ids": unsafe_enc["input_ids"],
            "unsafe_attention_mask": unsafe_enc["attention_mask"],
        }

    def _encode(self, prompt: str) -> dict[str, torch.Tensor]:
        """Tokenize a prompt."""
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        enc = tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


class SplitContrastDataset(Dataset):
    """Single-side view of a ContrastDataset (safe or unsafe only).

    This is used to create separate DataLoaders for safe and unsafe inputs
    for the subspace identifier, which expects separate DataLoaders.

    Args:
        contrast_dataset: The parent ContrastDataset.
        side: ``"safe"`` or ``"unsafe"``.
    """

    def __init__(self, contrast_dataset: ContrastDataset, side: str) -> None:
        if side not in ("safe", "unsafe"):
            raise ValueError(f"side must be 'safe' or 'unsafe', got {side!r}")
        self._parent = contrast_dataset
        self._side = side

    def __len__(self) -> int:
        return len(self._parent)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self._parent[idx]
        return {
            "input_ids": item[f"{self._side}_input_ids"],
            "attention_mask": item[f"{self._side}_attention_mask"],
        }


class RDSATrainDataset(Dataset):
    """Mixed training dataset for RDSA fine-tuning.

    Combines harmful (e.g. StrongReject) and benign (e.g. VQAv2) samples.
    Each sample includes an ``is_harmful`` flag for conditional loss computation.

    Args:
        harmful_samples: List of dicts with ``"prompt"`` and ``"response"`` keys.
        benign_samples: List of dicts with ``"prompt"`` and ``"response"`` keys.
        processor: HuggingFace processor for tokenization.
        ratio: Target harmful-to-benign ratio. ``1.0`` means equal counts.
        max_length: Maximum sequence length.
        seed: Random seed for reproducible shuffling.
    """

    def __init__(
        self,
        harmful_samples: list[dict[str, str]],
        benign_samples: list[dict[str, str]],
        processor: Any,
        ratio: float = 1.0,
        max_length: int = 512,
        seed: int = 42,
    ) -> None:
        self.processor = processor
        self.max_length = max_length

        # Balance datasets according to ratio
        rng = random.Random(seed)

        n_benign = len(benign_samples)
        n_harmful = min(len(harmful_samples), int(n_benign * ratio))

        # Subsample harmful if needed
        if n_harmful < len(harmful_samples):
            harmful_samples = rng.sample(harmful_samples, n_harmful)

        self.samples: list[tuple[dict[str, str], bool]] = []
        for s in harmful_samples:
            self.samples.append((s, True))
        for s in benign_samples[:n_benign]:
            self.samples.append((s, False))

        rng.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample, is_harmful = self.samples[idx]

        prompt = sample["prompt"]
        response = sample.get("response", "")

        tokenizer = getattr(self.processor, "tokenizer", self.processor)

        # Encode prompt + response as a sequence
        text = f"{prompt}\n{response}" if response else prompt
        enc = tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "is_harmful": torch.tensor(is_harmful, dtype=torch.bool),
        }


class SafetyEvalDataset(Dataset):
    """Dataset for attack evaluation using AdvBench or similar benchmarks.

    Loads harmful instructions from a CSV file (one instruction per row,
    first column is the instruction).

    Args:
        data_path: Path to CSV file with harmful instructions.
        processor: HuggingFace processor for tokenization.
        max_length: Maximum sequence length.
    """

    def __init__(
        self,
        data_path: str,
        processor: Any,
        max_length: int = 512,
    ) -> None:
        self.processor = processor
        self.max_length = max_length
        self.instructions: list[str] = []

        with open(data_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    self.instructions.append(row[0].strip())

    def __len__(self) -> int:
        return len(self.instructions)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        instruction = self.instructions[idx]

        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        enc = tokenizer(
            instruction,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "instruction": instruction,
        }


# ------------------------------------------------------------------
# DataLoader factory functions
# ------------------------------------------------------------------


def create_contrast_dataloaders(
    safe_path: str,
    unsafe_path: str,
    processor: Any,
    batch_size: int = 16,
    num_workers: int = 0,
    max_length: int = 512,
) -> tuple[DataLoader, DataLoader]:
    """Create separate safe and unsafe DataLoaders for subspace identification.

    Args:
        safe_path: Path to safe prompts JSONL file.
        unsafe_path: Path to unsafe prompts JSONL file.
        processor: HuggingFace processor.
        batch_size: Batch size for both loaders.
        num_workers: Number of data loading workers.
        max_length: Maximum sequence length.

    Returns:
        ``(safe_dataloader, unsafe_dataloader)`` tuple.
    """
    contrast_ds = ContrastDataset(safe_path, unsafe_path, processor, max_length)

    safe_ds = SplitContrastDataset(contrast_ds, "safe")
    unsafe_ds = SplitContrastDataset(contrast_ds, "unsafe")

    safe_dl = DataLoader(
        safe_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    unsafe_dl = DataLoader(
        unsafe_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    return safe_dl, unsafe_dl


def create_train_dataloader(
    harmful_samples: list[dict[str, str]],
    benign_samples: list[dict[str, str]],
    processor: Any,
    batch_size: int = 4,
    num_workers: int = 0,
    ratio: float = 1.0,
    max_length: int = 512,
    seed: int = 42,
) -> DataLoader:
    """Create a training DataLoader with mixed harmful/benign samples.

    Args:
        harmful_samples: List of harmful sample dicts.
        benign_samples: List of benign sample dicts.
        processor: HuggingFace processor.
        batch_size: Training batch size.
        num_workers: Number of data loading workers.
        ratio: Harmful-to-benign ratio.
        max_length: Maximum sequence length.
        seed: Random seed.

    Returns:
        Training DataLoader.
    """
    dataset = RDSATrainDataset(
        harmful_samples=harmful_samples,
        benign_samples=benign_samples,
        processor=processor,
        ratio=ratio,
        max_length=max_length,
        seed=seed,
    )
    return DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
