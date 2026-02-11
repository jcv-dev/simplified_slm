# -*- coding: utf-8 -*-

"""
Fused RMSNorm with Swish Gate.

Combines normalization and gating in a single operation for efficiency.
"""

import torch
import torch.nn as nn


class FusedRMSNormSwishGate(nn.Module):
    """
    Fused RMSNorm + Swish Gate operation.
    
    Computes: RMSNorm(x) * gate * sigmoid(gate)
    
    This combines:
    1. RMS normalization on input x
    2. Swish gating with separate gate signal
    
    Args:
        hidden_size: Size of hidden dimension
        eps: Small constant for numerical stability
    """
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """
        Apply fused RMSNorm + Swish gate.
        
        Args:
            x: Input tensor to normalize
            gate: Gate signal for swish gating
            
        Returns:
            Normalized and gated output
        """
        # RMSNorm
        variance = x.pow(2).mean(-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps)
        x_norm = self.weight * x_norm
        
        # Swish gate: output * gate * sigmoid(gate)
        return x_norm * gate * torch.sigmoid(gate)
