# -*- coding: utf-8 -*-

"""
Optimizer and learning rate scheduler builders.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from simplified_slm.training.config import TrainingConfig


def build_optimizer(
    model_parameters: Iterable[torch.nn.Parameter],
    config: TrainingConfig,
) -> AdamW:
    """
    Build AdamW optimizer with weight decay.
    
    Separates parameters into decay and no-decay groups.
    Bias, LayerNorm, and RMSNorm parameters don't get weight decay.
    
    Args:
        model_parameters: Model named parameters
        config: Training configuration
        
    Returns:
        Configured AdamW optimizer
    """
    decay_params = []
    no_decay_params = []
    
    for name, param in model_parameters:
        if not param.requires_grad:
            continue
        if param.dim() < 2 or 'bias' in name or 'norm' in name or 'pad_dimension' in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]
    
    optimizer = AdamW(
        param_groups,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
    )
    return optimizer


def build_scheduler(
    optimizer: AdamW,
    config: TrainingConfig,
) -> LambdaLR:
    """
    Build learning rate scheduler with warmup.
    
    Supports:
    - 'cosine': Cosine annealing with linear warmup
    - 'linear': Linear decay with linear warmup
    
    Args:
        optimizer: The optimizer
        config: Training configuration
        
    Returns:
        LambdaLR scheduler
    """
    warmup_steps = config.warmup_steps
    max_steps = config.max_steps

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        
        # Decay phase
        progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        
        if config.lr_scheduler == 'cosine':
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        elif config.lr_scheduler == 'linear':
            return max(0.0, 1.0 - progress)
        else:
            return 1.0

    return LambdaLR(optimizer, lr_lambda)
