# -*- coding: utf-8 -*-

"""
Main training loop for SimplifiedSLM and HNetBit models.

Supports:
- Gradient accumulation
- Mixed precision (FP16/BF16) via AMP
- Gradient clipping
- Checkpointing with optimizer state
- Periodic evaluation
"""

from __future__ import annotations

import math
import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from simplified_slm.training.config import TrainingConfig
from simplified_slm.training.data import collate_fn
from simplified_slm.training.optimizer import build_optimizer, build_optimizer_hierarchical, build_scheduler
from simplified_slm.training.logger import TrainingLogger


class Trainer:
    """
    Training loop for byte-level language models.
    
    Handles gradient accumulation, mixed precision, checkpointing, and evaluation.
    
    Args:
        model: The language model (SimplifiedSLMForCausalLM or HNetBitForCausalLM)
        config: Training configuration
        train_dataset: Training dataset
        val_dataset: Optional validation dataset
        logger: Optional training logger (created if None)
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataset,
        val_dataset=None,
        logger: Optional[TrainingLogger] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Device
        self.device = next(model.parameters()).device
        
        # Optimizer and scheduler (use hierarchical optimizer if lr_multipliers specified)
        if hasattr(config, 'lr_multipliers') and config.lr_multipliers:
            self.optimizer = build_optimizer_hierarchical(model, config)
        else:
            self.optimizer = build_optimizer(model.named_parameters(), config)
        self.scheduler = build_scheduler(self.optimizer, config)
        
        # Mixed precision
        self.use_amp = config.fp16 or config.bf16
        self.amp_dtype = torch.bfloat16 if config.bf16 else torch.float16
        self.scaler = torch.amp.GradScaler('cuda', enabled=config.fp16)
        
        # Logger
        self.logger = logger or TrainingLogger(config.output_dir)
        
        # State
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Load balancing for hierarchical models (HNetBit)
        self.use_load_balancing = (
            hasattr(config, 'lambda_lb') and config.lambda_lb > 0.0 and
            hasattr(model, 'backbone') and hasattr(model.backbone, '_apply_lr_multiplier')
        )
        self.lambda_lb = getattr(config, 'lambda_lb', 0.0)
        
        # DataLoader
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=0,
        )
        
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                drop_last=False,
                num_workers=0,
            )
        else:
            self.val_loader = None

    def compute_load_balancing_loss(self, router_outputs):
        """
        Compute load balancing loss for hierarchical models with dynamic chunking.
        
        Encourages balanced chunking across sequences to prevent degenerate solutions
        (all chunks or no chunks). Based on H-Net's load balancing loss.
        
        Args:
            router_outputs: List of RoutingModuleOutput from each hierarchy stage
        
        Returns:
            Load balancing loss tensor
        """
        if not router_outputs:
            return 0.0
        
        total_lb_loss = 0.0
        N = getattr(self.config, 'downsampling_factor', 2.5)
        
        for router_output in router_outputs:
            if router_output is None:
                continue
                
            # Get boundary predictions
            boundary_prob = router_output.boundary_prob
            tokenized_prob = boundary_prob[..., -1]  # Last token probabilities
            boundary_mask = router_output.boundary_mask
            
            # Compute ratios
            true_ratio = boundary_mask.float().mean()
            average_prob = tokenized_prob.float().mean()
            
            # Load balancing loss formula from H-Net
            stage_lb_loss = (
                (1 - true_ratio) * (1 - average_prob) +
                (true_ratio) * (average_prob) * (N - 1)
            ) * N / (N - 1)
            
            total_lb_loss += stage_lb_loss
        
        return total_lb_loss

    def train(self) -> None:
        """Run the full training loop."""
        self.model.train()
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        self.config.save(os.path.join(self.config.output_dir, "training_config.json"))
        
        # Resume if needed
        if self.config.resume_from:
            self._load_checkpoint(self.config.resume_from)
        
        print(f"Starting training: {self.config.max_steps} steps, "
              f"batch_size={self.config.batch_size}, "
              f"grad_accum={self.config.gradient_accumulation_steps}")
        
        accumulation_loss = 0.0
        micro_step = 0
        data_iter = iter(self.train_loader)
        
        while self.global_step < self.config.max_steps:
            # Get next batch (cycle through data)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)
            
            # Move to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass with optional AMP
            with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    labels=batch['labels'],
                    output_hidden_states=self.use_load_balancing,  # Get router outputs if needed
                )
                ce_loss = outputs.loss
                
                # Add load balancing loss for hierarchical models
                if self.use_load_balancing and outputs.hidden_states:
                    lb_loss = self.compute_load_balancing_loss(outputs.hidden_states)
                    total_loss = ce_loss + self.lambda_lb * lb_loss
                else:
                    total_loss = ce_loss
                    lb_loss = 0.0
                
                loss = total_loss / self.config.gradient_accumulation_steps
            
            # Backward
            self.scaler.scale(loss).backward()
            accumulation_loss += loss.item()
            micro_step += 1
            
            # Optimizer step after accumulation
            if micro_step % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm,
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                
                self.global_step += 1
                
                # Log
                if self.global_step % self.config.log_interval == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    log_dict = {
                        'loss': accumulation_loss,
                        'lr': lr,
                        'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    }
                    if self.use_load_balancing:
                        log_dict['lb_loss'] = lb_loss.item() if isinstance(lb_loss, torch.Tensor) else lb_loss
                    self.logger.log(log_dict, self.global_step)
                
                accumulation_loss = 0.0
                
                # Evaluate
                if (self.val_loader is not None and 
                    self.global_step % self.config.eval_interval == 0):
                    val_loss = self.evaluate()
                    self.logger.log_eval({'loss': val_loss}, self.global_step)
                    
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint("best")
                    
                    self.model.train()
                
                # Save checkpoint
                if self.global_step % self.config.save_interval == 0:
                    self._save_checkpoint(f"step_{self.global_step}")
        
        # Final save
        self._save_checkpoint("final")
        self.logger.close()
        print(f"Training complete. Best val loss: {self.best_val_loss:.4f}")

    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Run evaluation on validation set.
        
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    labels=batch['labels'],
                )
            
            total_loss += outputs.loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)

    def _save_checkpoint(self, name: str) -> None:
        """Save a training checkpoint."""
        path = os.path.join(self.config.output_dir, f"checkpoint_{name}.pt")
        
        # Get config dict
        model_config = self.model.config
        if hasattr(model_config, 'to_dict'):
            config_dict = model_config.to_dict()
        else:
            config_dict = {k: v for k, v in model_config.__dict__.items() 
                          if not k.startswith('_')}
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': config_dict,
            'training_config': self.config.__dict__,
        }, path)
        print(f"  Saved checkpoint: {path}")

    def _load_checkpoint(self, path: str) -> None:
        """Load a training checkpoint."""
        print(f"Resuming from {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"  Resumed at step {self.global_step}, best_val_loss={self.best_val_loss:.4f}")
