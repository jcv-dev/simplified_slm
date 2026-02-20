# -*- coding: utf-8 -*-

"""
Training configuration dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import json


@dataclass
class TrainingConfig:
    """
    Training hyperparameters and settings.
    
    Args:
        output_dir: Directory for checkpoints and logs
        data_path: Path to training data (text file or directory)
        val_data_path: Path to validation data (optional)
        max_steps: Maximum training steps
        batch_size: Per-device batch size
        gradient_accumulation_steps: Number of accumulation steps
        learning_rate: Peak learning rate
        weight_decay: AdamW weight decay
        max_grad_norm: Gradient clipping norm
        warmup_steps: Linear warmup steps
        lr_scheduler: Learning rate schedule ('cosine' or 'linear')
        max_seq_length: Maximum sequence length in bytes
        eval_interval: Steps between evaluations
        save_interval: Steps between checkpoint saves
        log_interval: Steps between log prints
        seed: Random seed
        fp16: Use FP16 mixed precision
        bf16: Use BF16 mixed precision
        resume_from: Path to checkpoint to resume from
    """
    
    # Paths
    output_dir: str = "./checkpoints"
    data_path: str = ""
    val_data_path: Optional[str] = None
    
    # Training schedule
    max_steps: int = 10000
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    
    # Optimizer
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 500
    lr_scheduler: str = "cosine"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    
    # Data
    max_seq_length: int = 512
    
    # Evaluation & logging
    eval_interval: int = 500
    save_interval: int = 1000
    log_interval: int = 10
    
    # Reproducibility
    seed: int = 42
    
    # Mixed precision
    fp16: bool = False
    bf16: bool = False
    
    # Resume
    resume_from: Optional[str] = None
    
    def save(self, path: str) -> None:
        """Save config to JSON."""
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """Load config from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
