# -*- coding: utf-8 -*-

"""
Dynamic Chunking module with ternary weights.

Adapted from H-Net's dc.py, replacing nn.Linear projections with BitLinear
for ternary weight quantization. Cosine similarity computation stays in FP32
for numerical stability.

Components:
- RoutingModuleBit: Predicts chunk boundaries via cosine similarity (BitLinear projections)
- ChunkLayer: Groups tokens into variable-length chunks based on boundaries
- DeChunkLayer: Reconstructs full sequence from chunk representations using EMA
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from simplified_slm.ops.bitnet import BitLinear


@dataclass
class RoutingModuleOutput:
    """Output of the routing module.
    
    Attributes:
        boundary_prob: Probability of boundary at each position, shape [..., 2]
                       (prob_no_boundary, prob_boundary)
        boundary_mask: Binary mask of selected boundaries, shape [...]
        selected_probs: Probability of the selected action, shape [..., 1]
    """
    boundary_prob: torch.Tensor
    boundary_mask: torch.Tensor
    selected_probs: torch.Tensor


@dataclass
class RoutingModuleState:
    """Inference state for the routing module.
    
    Attributes:
        has_seen_tokens: Whether each batch element has processed any tokens yet
        last_hidden_state: Last hidden state per batch element for boundary prediction
    """
    has_seen_tokens: torch.Tensor   # (batch_size,)
    last_hidden_state: torch.Tensor  # (batch_size, d_model)


@dataclass
class DeChunkState:
    """Inference state for the dechunk layer.
    
    Attributes:
        last_value: Last output value per batch element for EMA
    """
    last_value: torch.Tensor  # (batch_size, d_model)


class RoutingModuleBit(nn.Module):
    """
    Predicts chunk boundaries using cosine similarity with BitLinear projections.
    
    Computes boundary probability between consecutive tokens:
        boundary_prob = (1 - cos_sim(q_proj(h_t), k_proj(h_{t+1}))) / 2
    
    High boundary probability indicates semantic dissimilarity → chunk boundary.
    
    Args:
        d_model: Hidden dimension
    """

    def __init__(self, d_model: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        
        # BitLinear projections for cosine similarity (ternary weights)
        self.q_proj_layer = BitLinear(d_model, d_model, bias=False)
        self.k_proj_layer = BitLinear(d_model, d_model, bias=False)

        # Initialize to near-identity for stable training start
        with torch.no_grad():
            self.q_proj_layer.weight.copy_(torch.eye(d_model))
            self.k_proj_layer.weight.copy_(torch.eye(d_model))
        self.q_proj_layer.weight._no_reinit = True
        self.k_proj_layer.weight._no_reinit = True

    def allocate_inference_cache(
        self, batch_size: int, max_seqlen: int, device, dtype=None
    ) -> RoutingModuleState:
        """Allocate inference state for step-by-step generation."""
        return RoutingModuleState(
            has_seen_tokens=torch.zeros(batch_size, device=device, dtype=torch.bool),
            last_hidden_state=torch.zeros(batch_size, self.d_model, device=device, dtype=dtype),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        inference_params: Optional[RoutingModuleState] = None,
    ) -> RoutingModuleOutput:
        """
        Compute chunk boundaries for a sequence.
        
        Args:
            hidden_states: Input tensor (B, L, D)
            mask: Valid token mask (B, L), True for valid positions
            inference_params: Optional state for prefill phase
            
        Returns:
            RoutingModuleOutput with boundary predictions
        """
        assert mask is not None, "Mask must be provided"

        if inference_params is not None:
            assert (~inference_params.has_seen_tokens).all(), \
                "Cannot have seen tokens when running forward (use step for incremental)"

        # Cosine similarity between consecutive tokens (computed in FP32 for stability)
        q = F.normalize(self.q_proj_layer(hidden_states[:, :-1]).float(), dim=-1)
        k = F.normalize(self.k_proj_layer(hidden_states[:, 1:]).float(), dim=-1)
        cos_sim = torch.einsum("b l d, b l d -> b l", q, k)
        
        # Boundary probability: high when consecutive tokens are dissimilar
        boundary_prob = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)

        # Force first token to always be a boundary
        PAD_PROB = 1.0
        boundary_prob = F.pad(boundary_prob, (1, 0), "constant", PAD_PROB)

        # Stack as [prob_no_boundary, prob_boundary]
        boundary_prob = torch.stack(((1 - boundary_prob), boundary_prob), dim=-1)

        # Select boundary or not (argmax → hard decision)
        selected_idx = torch.argmax(boundary_prob, dim=-1)
        boundary_mask = selected_idx == 1  # True where boundary

        # Mask out invalid positions
        if mask is not None:
            boundary_mask = boundary_mask & mask

        # Update inference state during prefill
        if inference_params is not None:
            has_mask = mask.any(dim=-1)
            inference_params.has_seen_tokens.copy_(
                has_mask | inference_params.has_seen_tokens
            )
            last_mask = torch.clamp(mask.sum(dim=-1) - 1, min=0)
            inference_params.last_hidden_state.copy_(
                torch.where(
                    has_mask.unsqueeze(-1).expand_as(inference_params.last_hidden_state),
                    hidden_states[
                        torch.arange(hidden_states.shape[0], device=hidden_states.device),
                        last_mask,
                    ],
                    inference_params.last_hidden_state,
                )
            )

        selected_probs = boundary_prob.gather(
            dim=-1, index=selected_idx.unsqueeze(-1)
        )

        return RoutingModuleOutput(
            boundary_prob=boundary_prob,
            boundary_mask=boundary_mask,
            selected_probs=selected_probs,
        )

    def step(
        self,
        hidden_states: torch.Tensor,
        inference_params: RoutingModuleState,
    ) -> RoutingModuleOutput:
        """
        Step-by-step boundary prediction for generation.
        
        Args:
            hidden_states: Current token hidden state (B, 1, D)
            inference_params: Current routing state
            
        Returns:
            RoutingModuleOutput for current step
        """
        hidden_states_squeezed = hidden_states.squeeze(1)  # (B, D)
        
        # Cosine similarity with last token (FP32 for stability)
        q = F.normalize(self.q_proj_layer(inference_params.last_hidden_state).float(), dim=-1)
        k = F.normalize(self.k_proj_layer(hidden_states_squeezed).float(), dim=-1)
        cos_sim = torch.einsum("b d, b d -> b", q, k)
        
        boundary_prob = torch.clamp(((1 - cos_sim) / 2), min=0.0, max=1.0)
        
        # Update state
        inference_params.last_hidden_state.copy_(hidden_states_squeezed)

        # First token is always a boundary
        boundary_prob = torch.where(
            inference_params.has_seen_tokens,
            boundary_prob,
            torch.ones_like(boundary_prob),
        )
        boundary_prob = torch.stack(((1 - boundary_prob), boundary_prob), dim=-1)

        inference_params.has_seen_tokens.copy_(
            torch.ones_like(inference_params.has_seen_tokens)
        )

        return RoutingModuleOutput(
            boundary_prob=boundary_prob,
            boundary_mask=boundary_prob[..., 1] > 0.5,
            selected_probs=boundary_prob.max(dim=-1).values.unsqueeze(-1),
        )


class ChunkLayer(nn.Module):
    """
    Groups tokens into variable-length chunks based on boundary mask.
    
    No learnable parameters — pure selection/gathering operation.
    Works in unpacked mode (B, L, D) with mask.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        boundary_mask: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Select boundary tokens to form chunks.
        
        Args:
            hidden_states: Input tensor (B, L, D)
            boundary_mask: Boolean boundary mask (B, L)
            mask: Valid token mask (B, L)
            
        Returns:
            Tuple of (next_hidden_states, next_mask):
                next_hidden_states: Chunked tokens (B, M, D) where M = max chunks
                next_mask: Valid chunk mask (B, M)
        """
        assert mask is not None, "Mask must be provided"
        
        num_tokens = boundary_mask.sum(dim=-1)  # (B,)
        next_max_seqlen = int(num_tokens.max())

        device = hidden_states.device
        L = hidden_states.shape[1]
        
        # Sort: boundary tokens first, then non-boundary
        token_idx = (
            torch.arange(L, device=device)[None, :] + (~boundary_mask).long() * L
        )
        seq_sorted_indices = torch.argsort(token_idx, dim=1)

        # Gather the first next_max_seqlen tokens (the boundary tokens)
        next_hidden_states = torch.gather(
            hidden_states,
            dim=1,
            index=seq_sorted_indices[:, :next_max_seqlen, None].expand(
                -1, -1, hidden_states.shape[-1]
            ),
        )

        next_mask = (
            torch.arange(next_max_seqlen, device=device)[None, :]
            < num_tokens[:, None]
        )

        return next_hidden_states, next_mask

    def step(
        self,
        hidden_states: torch.Tensor,
        boundary_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Step-by-step chunk selection for generation.
        
        Args:
            hidden_states: Current hidden states (B, 1, D)
            boundary_mask: Boundary decisions (B,)
            
        Returns:
            Selected hidden states (B', 1, D) where B' = sum(boundary_mask)
        """
        return hidden_states[boundary_mask]


class DeChunkLayer(nn.Module):
    """
    Reconstructs full sequence from chunk representations using EMA.
    
    Uses exponential moving average to propagate chunk information to all
    positions between boundaries. Implemented as a simple sequential EMA
    (no Mamba2 dependency).
    
    Args:
        d_model: Hidden dimension
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def allocate_inference_cache(
        self, batch_size: int, max_seqlen: int, device, dtype=None,
    ) -> DeChunkState:
        """Allocate inference state for generation."""
        return DeChunkState(
            last_value=torch.zeros(batch_size, self.d_model, device=device, dtype=dtype),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        boundary_mask: torch.Tensor,
        boundary_prob: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        inference_params: Optional[DeChunkState] = None,
    ) -> torch.Tensor:
        """
        Reconstruct full sequence from chunked representations.
        
        Uses EMA: out_t = p_t * chunk_t + (1 - p_t) * out_{t-1}
        where p_t is the boundary probability. At boundaries, the chunk
        value dominates; between boundaries, previous values are carried forward.
        
        Args:
            hidden_states: Chunked representations (B, M, D)
            boundary_mask: Original boundary mask (B, L)
            boundary_prob: Boundary probabilities (B, L, 2)
            mask: Valid token mask (B, L)
            inference_params: Optional state for prefill
            
        Returns:
            Reconstructed sequence (B, L, D)
        """
        B, L = boundary_mask.shape
        D = hidden_states.shape[-1]
        device = hidden_states.device
        original_dtype = hidden_states.dtype

        # Map each position to its corresponding chunk index
        plug_back_idx = torch.cumsum(boundary_mask, dim=1) - 1  # (B, L)
        
        # Gather chunk representations for all positions
        expanded = torch.gather(
            hidden_states,
            dim=1,
            index=plug_back_idx.unsqueeze(-1).expand(-1, -1, D),
        )  # (B, L, D)

        # EMA mixing weights from boundary probabilities
        p = torch.clamp(boundary_prob[..., -1].float(), min=1e-4, max=1 - 1e-4)  # (B, L)

        # Sequential EMA: out_t = p_t * expanded_t + (1 - p_t) * out_{t-1}
        # Use list + stack to avoid in-place ops that break autograd
        steps = []
        prev = torch.zeros(B, D, device=device, dtype=torch.float32)
        for t in range(L):
            if t == 0:
                cur = expanded[:, t].float()
            else:
                cur = p[:, t].unsqueeze(-1) * expanded[:, t].float() + \
                      (1 - p[:, t].unsqueeze(-1)) * prev
            steps.append(cur)
            prev = cur
        out = torch.stack(steps, dim=1)  # (B, L, D)

        if inference_params is not None:
            inference_params.last_value.copy_(out[:, -1].to(original_dtype))

        return out.to(original_dtype)

    def step(
        self,
        hidden_states: torch.Tensor,
        boundary_mask: torch.Tensor,
        boundary_prob: torch.Tensor,
        inference_params: DeChunkState,
    ) -> torch.Tensor:
        """
        Step-by-step dechunking for generation.
        
        Args:
            hidden_states: Chunk hidden states (B', 1, D) where B' = sum(boundary_mask)
            boundary_mask: Boundary decisions (B,)
            boundary_prob: Boundary probabilities (B, 2)
            inference_params: Current EMA state
            
        Returns:
            Reconstructed output (B, 1, D)
        """
        B = boundary_mask.shape[0]
        D = inference_params.last_value.shape[-1]
        device = inference_params.last_value.device
        dtype = inference_params.last_value.dtype

        p = torch.zeros(B, device=device, dtype=dtype)
        p[boundary_mask] = boundary_prob[boundary_mask, -1].clamp(min=1e-4, max=1 - 1e-4)

        current_hidden_states = torch.zeros(B, D, device=device, dtype=dtype)
        if hidden_states.shape[0] > 0:
            current_hidden_states[boundary_mask] = hidden_states.squeeze(1)

        result = p.unsqueeze(-1) * current_hidden_states + \
                 (1 - p.unsqueeze(-1)) * inference_params.last_value
        inference_params.last_value.copy_(result)

        return result.unsqueeze(1)
