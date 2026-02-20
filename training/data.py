# -*- coding: utf-8 -*-

"""
Dataset classes for byte-level language modeling.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


class ByteLevelDataset(Dataset):
    """
    Dataset that converts raw bytes into fixed-length sequences for training.
    
    Takes pre-encoded byte data (list of ints or tensor) and creates
    overlapping or non-overlapping chunks of max_seq_length.
    
    Args:
        data: Byte data as list of ints or 1D tensor
        max_seq_length: Length of each training sequence
        stride: Step between sequences (default = max_seq_length for no overlap)
    """

    def __init__(
        self,
        data: torch.Tensor,
        max_seq_length: int = 512,
        stride: Optional[int] = None,
    ):
        if isinstance(data, list):
            data = torch.tensor(data, dtype=torch.long)
        self.data = data
        self.max_seq_length = max_seq_length
        self.stride = stride or max_seq_length
        
        # +1 for the target shift
        self.num_samples = max(0, (len(self.data) - max_seq_length - 1) // self.stride + 1)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx * self.stride
        end = start + self.max_seq_length
        
        input_ids = self.data[start:end]
        labels = self.data[start + 1:end + 1]
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': torch.ones(len(input_ids), dtype=torch.bool),
        }


class TextFileDataset(Dataset):
    """
    Dataset that reads a text file and converts to byte sequences.
    
    Reads the entire file as UTF-8 bytes, then chunks into fixed-length sequences.
    
    Args:
        file_path: Path to text file
        max_seq_length: Length of each training sequence
        stride: Step between sequences
    """

    def __init__(
        self,
        file_path: str,
        max_seq_length: int = 512,
        stride: Optional[int] = None,
    ):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Convert to bytes
        byte_data = list(text.encode('utf-8'))
        self.inner = ByteLevelDataset(
            torch.tensor(byte_data, dtype=torch.long),
            max_seq_length=max_seq_length,
            stride=stride,
        )

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.inner[idx]


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate batch of samples, padding to max length in batch.
    
    Args:
        batch: List of sample dicts with 'input_ids', 'labels', 'attention_mask'
        
    Returns:
        Batched dict with padded tensors
    """
    max_len = max(sample['input_ids'].shape[0] for sample in batch)
    
    input_ids = []
    labels = []
    attention_masks = []
    
    for sample in batch:
        seq_len = sample['input_ids'].shape[0]
        pad_len = max_len - seq_len
        
        if pad_len > 0:
            input_ids.append(torch.cat([sample['input_ids'], torch.zeros(pad_len, dtype=torch.long)]))
            labels.append(torch.cat([sample['labels'], torch.full((pad_len,), -100, dtype=torch.long)]))
            attention_masks.append(torch.cat([sample['attention_mask'], torch.zeros(pad_len, dtype=torch.bool)]))
        else:
            input_ids.append(sample['input_ids'])
            labels.append(sample['labels'])
            attention_masks.append(sample['attention_mask'])
    
    return {
        'input_ids': torch.stack(input_ids),
        'labels': torch.stack(labels),
        'attention_mask': torch.stack(attention_masks),
    }


def create_synthetic_data(
    num_bytes: int = 100000,
    vocab_size: int = 256,
    seed: int = 42,
) -> torch.Tensor:
    """
    Create synthetic byte data for testing training pipeline.
    
    Args:
        num_bytes: Number of bytes to generate
        vocab_size: Vocabulary size (256 for byte-level)
        seed: Random seed
        
    Returns:
        1D tensor of random byte values
    """
    torch.manual_seed(seed)
    return torch.randint(0, vocab_size, (num_bytes,), dtype=torch.long)
