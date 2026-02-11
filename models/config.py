# -*- coding: utf-8 -*-

"""
Configuration for Simplified SLM with ternary weights.

Simplified from HGRNBitConfig with:
- use_short_conv=False
- use_lower_bound=False
- num_heads=1, expand_ratio=1
- vocab_size=256 (byte-level tokenization)
"""

from typing import Optional

from transformers.configuration_utils import PretrainedConfig


class SimplifiedSLMConfig(PretrainedConfig):
    """
    Configuration class for SimplifiedSLM model.
    
    Args:
        vocab_size: Vocabulary size (256 for byte-level)
        hidden_size: Hidden dimension
        num_hidden_layers: Number of transformer blocks
        attn_mode: Attention mode ('fused_recurrent')
        num_heads: Number of attention heads (1 for simplified)
        expand_ratio: State expansion ratio (1 for simplified)
        use_short_conv: Whether to use short convolution (False)
        conv_size: Convolution kernel size (unused)
        share_conv_kernel: Whether to share conv kernel (unused)
        use_lower_bound: Whether to use forget gate lower bound (False)
        hidden_ratio: MLP hidden ratio
        intermediate_size: Override MLP intermediate size
        hidden_act: Activation function
        max_position_embeddings: Maximum sequence length
        rms_norm_eps: Epsilon for RMSNorm
        use_cache: Whether to use KV cache
        initializer_range: Std for weight initialization
        fuse_cross_entropy: Whether to use fused cross entropy
    """

    model_type = 'simplified_slm'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        vocab_size: int = 256,  # Byte-level tokenization
        hidden_size: int = 512,
        num_hidden_layers: int = 6,
        attn_mode: str = "fused_recurrent",
        num_heads: Optional[int] = 1,
        expand_ratio: Optional[int] = 1,
        use_short_conv: bool = False,
        conv_size: int = 4,
        share_conv_kernel: bool = True,
        use_lower_bound: bool = False,
        hidden_ratio: Optional[int] = 4,
        intermediate_size: Optional[int] = None,
        hidden_act: str = "swish",
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int = None,
        bos_token_id: int = 254,  # Byte tokenizer BOS
        eos_token_id: int = 255,  # Byte tokenizer EOS
        tie_word_embeddings: bool = False,
        initializer_range: float = 0.02,
        fuse_cross_entropy: bool = True,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.attn_mode = attn_mode
        self.num_heads = num_heads
        self.expand_ratio = expand_ratio
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.share_conv_kernel = share_conv_kernel
        self.use_lower_bound = use_lower_bound
        self.hidden_ratio = hidden_ratio
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.fuse_cross_entropy = fuse_cross_entropy

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @classmethod
    def small(cls, **kwargs):
        """Small model configuration (~10M params)."""
        return cls(
            hidden_size=256,
            num_hidden_layers=4,
            hidden_ratio=4,
            **kwargs
        )

    @classmethod
    def base(cls, **kwargs):
        """Base model configuration (~50M params)."""
        return cls(
            hidden_size=512,
            num_hidden_layers=6,
            hidden_ratio=4,
            **kwargs
        )

    @classmethod
    def large(cls, **kwargs):
        """Large model configuration (~100M params)."""
        return cls(
            hidden_size=768,
            num_hidden_layers=12,
            hidden_ratio=4,
            **kwargs
        )
