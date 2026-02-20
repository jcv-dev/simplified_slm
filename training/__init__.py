# -*- coding: utf-8 -*-

"""Training infrastructure for simplified_slm."""

from simplified_slm.training.config import TrainingConfig
from simplified_slm.training.trainer import Trainer
from simplified_slm.training.data import ByteLevelDataset, TextFileDataset, collate_fn
from simplified_slm.training.optimizer import build_optimizer, build_scheduler
from simplified_slm.training.logger import TrainingLogger
from simplified_slm.training.dataset_loader import (
    DatasetConfig,
    DatasetStatistics,
    HuggingFaceDataset,
    TinyStoriesDataset,
    WikiTextDataset,
    TextDirectoryDataset,
    create_train_val_split,
    prepare_datasets,
    print_dataset_stats,
)
from simplified_slm.training.metrics import (
    EvaluationMetrics,
    compute_bpb,
    compute_perplexity,
    compute_accuracy,
    compute_utf8_validity,
    evaluate_model,
    evaluate_generation_quality,
    compute_model_flops,
    compute_memory_footprint,
    MetricsTracker,
)

__all__ = [
    # Config
    "TrainingConfig",
    "DatasetConfig",
    "DatasetStatistics",
    # Trainer
    "Trainer",
    # Datasets
    "ByteLevelDataset",
    "TextFileDataset",
    "HuggingFaceDataset",
    "TinyStoriesDataset",
    "WikiTextDataset",
    "TextDirectoryDataset",
    "collate_fn",
    "create_train_val_split",
    "prepare_datasets",
    "print_dataset_stats",
    # Metrics
    "EvaluationMetrics",
    "compute_bpb",
    "compute_perplexity",
    "compute_accuracy",
    "compute_utf8_validity",
    "evaluate_model",
    "evaluate_generation_quality",
    "compute_model_flops",
    "compute_memory_footprint",
    "MetricsTracker",
    # Optimizer
    "build_optimizer",
    "build_scheduler",
    # Logger
    "TrainingLogger",
]
