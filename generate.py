#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Text generation script for Simplified SLM.

Usage:
    python generate.py --prompt "Hello world" --max_tokens 100
    python generate.py --model_path ./checkpoints/model.pt --prompt "Once upon a time"
"""

import argparse
import json
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simplified_slm.models import SimplifiedSLMConfig, SimplifiedSLMForCausalLM
from simplified_slm.models.hnet_bit import HNetBitConfig, HNetBitForCausalLM
from simplified_slm.utils import ByteTokenizer


def load_model(
    model_path: str = None,
    config=None,
    device: str = 'cuda',
    model_type: str = 'flat',
):
    """
    Load or create a model (flat SimplifiedSLM or hierarchical HNetBit).
    
    Args:
        model_path: Path to saved model checkpoint
        config: Model configuration (used if no checkpoint)
        device: Device to load model on
        model_type: 'flat' for SimplifiedSLM, 'hierarchical' for HNetBit
        
    Returns:
        Loaded model
    """
    if model_path and os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if 'config' in checkpoint:
            cfg_dict = checkpoint['config']
            detected_type = cfg_dict.get('model_type', model_type)
            if detected_type == 'hnet_bit':
                config = HNetBitConfig(**cfg_dict)
                model_type = 'hierarchical'
            else:
                config = SimplifiedSLMConfig(**cfg_dict)
                model_type = 'flat'
        
        if model_type == 'hierarchical':
            config = config or HNetBitConfig()
            model = HNetBitForCausalLM(config)
        else:
            config = config or SimplifiedSLMConfig()
            model = SimplifiedSLMForCausalLM(config)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        if model_type == 'hierarchical':
            config = config or HNetBitConfig()
            print(f"Creating new HNetBit model: d_model={config.d_model}, stages={config.num_stages}")
            model = HNetBitForCausalLM(config)
        else:
            config = config or SimplifiedSLMConfig()
            print(f"Creating new model with config: {config.hidden_size}d, {config.num_hidden_layers}L")
            model = SimplifiedSLMForCausalLM(config)
    
    model = model.to(device)
    model.eval()
    return model


def generate_text(
    model: SimplifiedSLMForCausalLM,
    tokenizer: ByteTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    do_sample: bool = True,
    device: str = 'cuda'
) -> str:
    """
    Generate text from a prompt.
    
    Args:
        model: The language model
        tokenizer: Byte tokenizer
        prompt: Input prompt string
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter
        top_p: Nucleus sampling parameter
        do_sample: Whether to sample (False = greedy)
        device: Device for generation
        
    Returns:
        Generated text string
    """
    # Encode prompt
    encoded = tokenizer.encode([prompt])[0]['input_ids']
    input_ids = torch.tensor([encoded.tolist()], dtype=torch.long, device=device)
    
    # Create attention mask
    attention_mask = torch.ones_like(input_ids)
    
    # Generate
    with torch.no_grad():
        if do_sample:
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=0,
            )
        else:
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=0,
            )
    
    # Decode output
    output_tokens = output_ids[0].tolist()
    generated_text = tokenizer.decode(output_tokens, errors='replace')
    
    return generated_text


def main():
    parser = argparse.ArgumentParser(description='Generate text with Simplified SLM')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default="Hello",
                        help='Input prompt')
    parser.add_argument('--max_tokens', type=int, default=100,
                        help='Maximum new tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Nucleus sampling threshold')
    parser.add_argument('--greedy', action='store_true',
                        help='Use greedy decoding')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--hidden_size', type=int, default=512,
                        help='Model hidden size (if creating new)')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of layers (if creating new flat model)')
    parser.add_argument('--model_type', type=str, default='flat',
                        choices=['flat', 'hierarchical'],
                        help='Model type: flat (SimplifiedSLM) or hierarchical (HNetBit)')
    parser.add_argument('--model_config', type=str, default=None,
                        help='Path to model config JSON (for hierarchical models)')
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Create config
    if args.model_config:
        with open(args.model_config, 'r') as f:
            cfg_dict = json.load(f)
        if cfg_dict.get('model_type') == 'hnet_bit':
            config = HNetBitConfig(**cfg_dict)
            args.model_type = 'hierarchical'
        else:
            config = SimplifiedSLMConfig(**cfg_dict)
            args.model_type = 'flat'
    elif args.model_type == 'hierarchical':
        config = HNetBitConfig()
    else:
        config = SimplifiedSLMConfig(
            vocab_size=256,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_layers,
        )
    
    # Load model
    model = load_model(args.model_path, config, args.device, args.model_type)
    tokenizer = ByteTokenizer()
    
    print(f"\nModel: {model.count_parameters():,} parameters")
    print(f"Prompt: {args.prompt}")
    print("-" * 50)
    
    # Generate
    generated = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=not args.greedy,
        device=args.device
    )
    
    print(f"Generated:\n{generated}")
    print("-" * 50)


if __name__ == "__main__":
    main()
