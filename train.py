#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Training entry point for SimplifiedSLM and HNetBit models.

Usage:
    # Train flat model with defaults
    python -m simplified_slm.train --output_dir runs/flat_test

    # Train hierarchical model with config
    python -m simplified_slm.train \
        --model_type hierarchical \
        --model_config configs/hnet_bit_1stage.json \
        --training_config configs/training_small.json

    # Resume from checkpoint
    python -m simplified_slm.train \
        --model_type hierarchical \
        --resume_from runs/experiment/checkpoint_step_1000.pt
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from simplified_slm.training.config import TrainingConfig
from simplified_slm.training.data import ByteLevelDataset, TextFileDataset, create_synthetic_data
from simplified_slm.training.trainer import Trainer
from simplified_slm.training.logger import TrainingLogger


def build_model(model_type: str, model_config_path: str | None, device: torch.device):
    """Build model from type and config."""

    if model_type == "hierarchical":
        from simplified_slm.models.hnet_bit import HNetBitConfig, HNetBitForCausalLM

        if model_config_path:
            with open(model_config_path, 'r') as f:
                cfg = json.load(f)
            config = HNetBitConfig(**cfg)
        else:
            config = HNetBitConfig.small_1stage()
        
        model = HNetBitForCausalLM(config)
    else:
        from simplified_slm.models.config import SimplifiedSLMConfig
        from simplified_slm.models.modeling import SimplifiedSLMForCausalLM

        if model_config_path:
            with open(model_config_path, 'r') as f:
                cfg = json.load(f)
            config = SimplifiedSLMConfig(**cfg)
        else:
            config = SimplifiedSLMConfig()
        
        model = SimplifiedSLMForCausalLM(config)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_type}, Parameters: {n_params:,}")
    return model.to(device)


def build_datasets(config: TrainingConfig, max_seq_length: int):
    """Build training and optional validation dataset."""

    if config.data_path and os.path.exists(config.data_path):
        print(f"Loading data from {config.data_path}")
        full_ds = TextFileDataset(config.data_path, seq_length=max_seq_length)
        
        # 90/10 split
        n = len(full_ds)
        n_val = max(1, n // 10)
        n_train = n - n_val
        
        train_ds = torch.utils.data.Subset(full_ds, range(n_train))
        val_ds = torch.utils.data.Subset(full_ds, range(n_train, n))
    else:
        print("No data_path found — using synthetic data for testing")
        data = create_synthetic_data(num_tokens=max_seq_length * 500)
        train_tokens = data[:int(len(data) * 0.9)]
        val_tokens = data[int(len(data) * 0.9):]
        
        train_ds = ByteLevelDataset(train_tokens, seq_length=max_seq_length)
        val_ds = ByteLevelDataset(val_tokens, seq_length=max_seq_length)
    
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    return train_ds, val_ds


def main():
    parser = argparse.ArgumentParser(description="Train SimplifiedSLM / HNetBit models")
    
    # Model
    parser.add_argument("--model_type", type=str, default="flat",
                        choices=["flat", "hierarchical"],
                        help="Model architecture type")
    parser.add_argument("--model_config", type=str, default=None,
                        help="Path to model config JSON")
    
    # Training config
    parser.add_argument("--training_config", type=str, default=None,
                        help="Path to training config JSON")
    
    # Override common training params via CLI
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--fp16", action="store_true", default=False)
    
    args = parser.parse_args()
    
    # Load or create training config
    if args.training_config:
        config = TrainingConfig.load(args.training_config)
    else:
        config = TrainingConfig()
    
    # CLI overrides
    for key in ['output_dir', 'data_path', 'max_steps', 'batch_size', 'lr',
                'max_seq_length', 'resume_from', 'seed', 'bf16', 'fp16']:
        val = getattr(args, key, None)
        if val is not None:
            setattr(config, key, val)
    
    # Defaults
    if config.output_dir == "runs/experiment":
        config.output_dir = f"runs/{args.model_type}_experiment"
    
    # Seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        config.fp16 = False
        config.bf16 = False
    
    # Build model
    model = build_model(args.model_type, args.model_config, device)
    
    # Build datasets
    max_seq_length = config.max_seq_length
    train_ds, val_ds = build_datasets(config, max_seq_length)
    
    # Logger
    logger = TrainingLogger(config.output_dir)
    
    # Train
    trainer = Trainer(
        model=model,
        config=config,
        train_dataset=train_ds,
        val_dataset=val_ds,
        logger=logger,
    )
    trainer.train()


if __name__ == "__main__":
    main()
