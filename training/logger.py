# -*- coding: utf-8 -*-

"""
Training logger with console, CSV, and optional WandB support.
"""

from __future__ import annotations

import csv
import math
import os
import time
from typing import Dict, Optional


class TrainingLogger:
    """
    Logger for training metrics.
    
    Outputs to console and CSV file. Optionally logs to Weights & Biases.
    
    Args:
        output_dir: Directory for log files
        use_wandb: Whether to use WandB logging
        wandb_project: WandB project name
        wandb_run_name: WandB run name
    """

    def __init__(
        self,
        output_dir: str,
        use_wandb: bool = False,
        wandb_project: str = "simplified-slm",
        wandb_run_name: Optional[str] = None,
    ):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.csv_path = os.path.join(output_dir, "training_log.csv")
        self.csv_file = None
        self.csv_writer = None
        self._csv_initialized = False
        
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    dir=output_dir,
                )
            except ImportError:
                print("WARNING: wandb not installed, skipping WandB logging")
        
        self.start_time = time.time()
        self._step_start = time.time()

    def _init_csv(self, keys):
        """Initialize CSV writer with headers."""
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=keys)
        self.csv_writer.writeheader()
        self._csv_initialized = True

    def log(self, metrics: Dict[str, float], step: int) -> None:
        """
        Log metrics for a training step.
        
        Args:
            metrics: Dict of metric name → value
            step: Current training step
        """
        metrics['step'] = step
        metrics['elapsed_seconds'] = time.time() - self.start_time
        
        # Compute BPB (bits per byte) from loss
        if 'loss' in metrics:
            metrics['bpb'] = metrics['loss'] / math.log(2)
            metrics['perplexity'] = math.exp(min(metrics['loss'], 20))  # Cap to avoid overflow
        
        # Console output
        parts = [f"step={step}"]
        for k, v in metrics.items():
            if k in ('step', 'elapsed_seconds'):
                continue
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        print(" | ".join(parts))
        
        # CSV
        if not self._csv_initialized:
            self._init_csv(list(metrics.keys()))
        self.csv_writer.writerow(metrics)
        self.csv_file.flush()
        
        # WandB
        if self.wandb_run is not None:
            import wandb
            wandb.log(metrics, step=step)

    def log_eval(self, metrics: Dict[str, float], step: int) -> None:
        """Log evaluation metrics."""
        print(f"  [EVAL] step={step} | " + " | ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        ))
        
        eval_metrics = {f"eval_{k}": v for k, v in metrics.items()}
        eval_metrics['step'] = step
        
        if self.wandb_run is not None:
            import wandb
            wandb.log(eval_metrics, step=step)

    def close(self) -> None:
        """Clean up resources."""
        if self.csv_file is not None:
            self.csv_file.close()
        if self.wandb_run is not None:
            import wandb
            wandb.finish()
