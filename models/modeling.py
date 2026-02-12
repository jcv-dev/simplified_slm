# -*- coding: utf-8 -*-

"""
Simplified SLM Model with ternary weights.

Full model implementation combining:
- Byte-level embedding
- N × HGRNBitBlock layers
- RMSNorm final normalization
- BitLinear LM head

Adapted from matmulfreellm HGRNBitForCausalLM.
"""

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from simplified_slm.models.config import SimplifiedSLMConfig
from simplified_slm.layers.hgrn_bit import HGRNBitBlock
from simplified_slm.ops.bitnet import BitLinear, RMSNorm
from simplified_slm.utils.cache import RecurrentCache


logger = logging.get_logger(__name__)


class SimplifiedSLMPreTrainedModel(PreTrainedModel):
    """Base class for SimplifiedSLM models."""

    config_class = SimplifiedSLMConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ['HGRNBitBlock']

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(
        self,
        module: nn.Module,
        rescale_prenorm_residual: bool = True,
        num_residuals_per_layer: int = 2,
    ):
        """Initialize weights with scaled initialization for residual layers."""
        if isinstance(module, (nn.Linear, nn.Conv1d, BitLinear)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        if rescale_prenorm_residual:
            # Scale residual layer weights by 1/sqrt(2 * n_layers)
            # Based on GPT-2 paper recommendations
            for name, p in module.named_parameters():
                if name in ["o_proj.weight", "down_proj.weight"]:
                    with torch.no_grad():
                        p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)


class SimplifiedSLMModel(SimplifiedSLMPreTrainedModel):
    """
    SimplifiedSLM base model (without LM head).
    
    Architecture:
        Embedding → N×HGRNBitBlock → RMSNorm
    """

    def __init__(self, config: SimplifiedSLMConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Token embeddings
        self.embeddings = nn.Embedding(
            config.vocab_size, 
            config.hidden_size, 
            self.padding_idx
        )
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            HGRNBitBlock(config, layer_idx) 
            for layer_idx in range(config.num_hidden_layers)
        ])
        
        # Final normalization
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            inputs_embeds: Optional pre-computed embeddings
            past_key_values: Optional cache for generation
            use_cache: Whether to return cache
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return ModelOutput
            
        Returns:
            BaseModelOutputWithPast or tuple
        """
        if output_attentions:
            warnings.warn("`SimplifiedSLMModel` does not support `output_attentions`, setting to False.")
            output_attentions = False
            
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Get input embeddings
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("Must specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds

        # Initialize cache
        if use_cache:
            if past_key_values is None:
                past_key_values = [layer.attn.init_state(batch_size) for layer in self.layers]
            if not isinstance(past_key_values, RecurrentCache):
                past_key_values = RecurrentCache.from_legacy_cache(past_key_values)

        # Disable cache during gradient checkpointing
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None

        # Process through layers
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                hidden_states, attentions, past_key_values = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    past_key_values,
                    use_cache,
                    output_attentions,
                    None  # lower_bound (disabled)
                )
            else:
                hidden_states, attentions, past_key_values = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    lower_bound=None  # disabled
                )

            if output_attentions:
                all_attns += (attentions,)

        # Final normalization
        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Convert cache format
        next_cache = None
        if use_cache:
            next_cache = past_key_values.to_legacy_cache()
            
        if not return_dict:
            return tuple(x for x in [hidden_states, next_cache, all_hidden_states, all_attns] if x is not None)
            
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attns
        )


class SimplifiedSLMForCausalLM(SimplifiedSLMPreTrainedModel, GenerationMixin):
    """
    SimplifiedSLM with Language Modeling head for causal LM.
    
    Architecture:
        SimplifiedSLMModel → BitLinear LM Head
    """
    
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: SimplifiedSLMConfig):
        super().__init__(config)
        self.model = SimplifiedSLMModel(config)
        self.vocab_size = config.vocab_size
        
        # LM head with ternary weights
        self.lm_head = BitLinear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def generate(self, *args, **kwargs):
        """Generate text with error handling for unsupported strategies."""
        try:
            return super().generate(*args, **kwargs)
        except AttributeError as exception:
            if 'past_key_values' in str(exception):
                raise AttributeError(
                    f"You tried to call `generate` with a decoding strategy that manipulates `past_key_values`, "
                    f"which is not supported for {self.__class__.__name__}. "
                    f"Try another generation strategy instead. "
                    f"For available strategies, see: "
                    f"https://huggingface.co/docs/transformers/en/generation_strategies#decoding-strategies"
                )
            else:
                raise exception

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """Prepare inputs for generation step."""
        # Convert to RecurrentCache if needed
        if past_key_values is not None and not isinstance(past_key_values, RecurrentCache):
            past_key_values = RecurrentCache.from_legacy_cache(past_key_values, input_ids.shape[1] - 1)
        
        # Only use last token if cache exists and has content
        if past_key_values is not None and len(past_key_values) > 0:
            input_ids, attention_mask = input_ids[:, -1:], attention_mask[:, -1:]
            
        # Use embeddings only for first step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            model_inputs = {'input_ids': input_ids.contiguous()}

        model_inputs.update({
            'past_key_values': past_key_values,
            'use_cache': kwargs.get('use_cache'),
            'attention_mask': attention_mask,
        })
        return model_inputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[List[torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        Forward pass for causal language modeling.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            inputs_embeds: Optional pre-computed embeddings
            past_key_values: Optional cache
            labels: Target labels for loss computation
            use_cache: Whether to return cache
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return ModelOutput
            
        Returns:
            CausalLMOutputWithPast or tuple
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Forward through base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Shift labels for next token prediction
            labels = labels.to(logits.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], loss_fct.ignore_index)), 1)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
            
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def get_ternary_weight_stats(self) -> dict:
        """Get statistics about ternary weight distribution."""
        stats = {'total_params': 0, 'ternary_params': 0, 'distribution': {-1: 0, 0: 0, 1: 0}}
        
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                from simplified_slm.ops.bitnet import weight_quant
                w_quant = weight_quant(param.data)
                scale = 1.0 / param.data.abs().mean().clamp_(min=1e-5)
                w_ternary = (param.data * scale).round().clamp_(-1, 1)
                
                stats['total_params'] += param.numel()
                stats['ternary_params'] += param.numel()
                stats['distribution'][-1] += (w_ternary == -1).sum().item()
                stats['distribution'][0] += (w_ternary == 0).sum().item()
                stats['distribution'][1] += (w_ternary == 1).sum().item()
        
        # Convert to percentages
        total = sum(stats['distribution'].values())
        if total > 0:
            stats['distribution_pct'] = {
                k: v / total * 100 for k, v in stats['distribution'].items()
            }
        
        return stats
