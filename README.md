# Simplified MatMul-Free SLM with Ternary Weights

A minimal implementation of a Small Language Model (SLM) using:
- **BitLinear** layers with ternary weight quantization ({-1, 0, +1})
- **HGRN** (Hierarchical Gated Recurrent Network) for efficient sequence modeling
- **Chunk-parallel** recurrence for fast training
- **Byte-level tokenization** (vocab_size=256)

## Architecture

```
Embedding (256 → hidden_size)
    ↓
N × HGRNBitBlock:
    ├── RMSNorm → HGRNBitAttention → Residual
    └── RMSNorm → HGRNBitMLP (SwiGLU) → Residual
    ↓
RMSNorm
    ↓
BitLinear LM Head (hidden_size → 256)
```

### Key Features

- **Ternary Weights**: All linear layers use weights quantized to {-1, 0, +1}
- **No MatMul**: Ternary weights enable efficient addition-based computation
- **Linear Complexity**: HGRN has O(n) complexity vs O(n²) for attention
- **Memory Efficient**: ~8x memory reduction vs FP16 weights (theoretical)

## Installation

```bash
# Requirements
pip install torch transformers triton==2.2 einops

# Install package
cd simplified-slm
pip install -e .
```

## Quick Start

```python
from simplified_slm.models import SimplifiedSLMConfig, SimplifiedSLMForCausalLM
from simplified_slm.utils import ByteTokenizer

# Create model
config = SimplifiedSLMConfig(
    vocab_size=256,
    hidden_size=512,
    num_hidden_layers=6,
    num_heads=1,
)
model = SimplifiedSLMForCausalLM(config).cuda()

# Tokenize
tokenizer = ByteTokenizer()
text = "Hello world"
encoded = tokenizer.encode([text])[0]['input_ids']
input_ids = torch.tensor([encoded.tolist()]).cuda()

# Generate
output = model.generate(input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0].tolist()))
```

## Model Configurations

| Config | Hidden | Layers | Params |
|--------|--------|--------|--------|
| Small  | 256    | 4      | ~10M   |
| Base   | 512    | 6      | ~50M   |
| Large  | 768    | 12     | ~100M  |

```python
# Use preset configs
config = SimplifiedSLMConfig.small()  # or .base() or .large()
```

## Project Structure

```
simplified-slm/
├── models/
│   ├── config.py           # SimplifiedSLMConfig
│   ├── modeling.py         # SimplifiedSLMForCausalLM
│   └── hnet_bit.py         # HNetBit hierarchical model
├── layers/
│   └── hgrn_bit.py         # HGRNBitAttention, HGRNBitMLP, HGRNBitBlock
├── ops/
│   ├── bitnet.py           # BitLinear, weight_quant, activation_quant
│   ├── activations.py      # SwiGLU, swish
│   ├── dynamic_chunking.py # Cosine-similarity based chunking
│   ├── fused_norm_gate.py  # FusedRMSNormSwishGate
│   └── hgrn/
│       ├── chunk.py          # Chunk-parallel HGRN
│       └── recurrent_fuse.py # Fused recurrent HGRN
├── training/
│   ├── trainer.py          # Training loop
│   ├── config.py           # TrainingConfig
│   ├── data.py             # ByteLevelDataset
│   ├── dataset_loader.py   # HuggingFace dataset integration
│   ├── metrics.py          # BPB, perplexity, evaluation metrics
│   ├── evaluator.py        # Comprehensive evaluation pipeline
│   ├── logger.py           # TensorBoard, W&B, CSV logging
│   └── optimizer.py        # Optimizer factory
├── utils/
│   ├── tokenizers.py       # ByteTokenizer
│   ├── cache.py            # RecurrentCache
│   ├── hnet_cache.py       # HNetBit cache
│   └── helpers.py          # Utility functions
├── scripts/
│   ├── prepare_tinystories.py  # Dataset preparation
│   ├── evaluate_model.py       # Model evaluation
│   ├── analyze_training.py     # Training analysis
│   └── run_experiment.py       # End-to-end pipeline
├── configs/
│   ├── slm_base.json                   # Flat model config
│   ├── hnet_bit_2stage.json            # 2-stage hierarchical
│   ├── training_2stage_tinystories.json # 2-stage training config
│   └── training_flat_tinystories.json  # Flat training config
├── tests/
│   ├── test_model.py           # Model tests
│   ├── test_dataset_loader.py  # Dataset loader tests
│   ├── test_metrics.py         # Metrics tests
│   └── test_evaluator.py       # Evaluator tests
├── docs/
│   └── TRAINING_GUIDE.md       # Comprehensive training guide
├── train.py                    # Training entry point
└── generate.py                 # Text generation script
```

## Technical Details

### Ternary Weight Quantization

```python
def weight_quant(w):
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    return (w * scale).round().clamp_(-1, 1) / scale
```

Uses Straight-Through Estimator (STE) for gradient flow:
```python
w_quant = w + (weight_quant(w) - w).detach()
```

### HGRN Recurrence

```
f_t = sigmoid(f_proj(x_t))           # Forget gate
i_t = swiglu(i_proj(x_t), 1 - f_t)   # Input with complement gate
h_t = f_t * h_{t-1} + i_t            # State update
o_t = g_norm(g_proj(x_t), h_t)       # Gated output
```

### Chunk-Parallel Training

Uses Triton kernels for parallelized recurrence computation:
- Forward: O(T/chunk_size) parallel chunks
- Backward: Gradient accumulation across chunks
- Much faster than sequential for long sequences

## Running Tests

```bash
cd simplified-slm
python -m pytest tests/ -v

# Or run directly
python tests/test_model.py
```

## Generation

```bash
# Basic generation
python generate.py --prompt "Hello" --max_tokens 100 --temperature 0.8
python generate.py --model_path ./checkpoint.pt --prompt "Once upon a time" --greedy

# Interactive mode
python generate.py --model_path ./checkpoint.pt --interactive

# Batch generation from file
python generate.py --model_path ./checkpoint.pt --prompt_file prompts.txt --output_file outputs.txt
```

## Training

Full training infrastructure is now available! See [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) for comprehensive documentation.

### Quick Start

```bash
# 1. Prepare TinyStories dataset
python scripts/prepare_tinystories.py --output_dir ./data/tinystories

# 2. Train 2-stage hierarchical model
python train.py --config configs/training_2stage_tinystories.json

# 3. Evaluate model
python scripts/evaluate_model.py --checkpoint ./checkpoints/best_model.pt --dataset tinystories

# 4. Generate text
python generate.py --model_path ./checkpoints/best_model.pt --prompt "Once upon a time"
```

### End-to-End Pipeline

```bash
# Run complete experiment (data → train → evaluate → analyze)
python scripts/run_experiment.py --config configs/training_2stage_tinystories.json
```

### Training Features

- **Datasets**: TinyStories, WikiText-2/103, custom text directories
- **Mixed Precision**: BF16 training with PyTorch AMP
- **Logging**: TensorBoard + Weights & Biases + CSV export
- **Metrics**: BPB (bits per byte), perplexity, accuracy, UTF-8 validity
- **Monitoring**: Ternary weight stats, gradient norms, generation samples
- **Analysis**: Training curves, loss visualization, checkpoint comparison

### Model Configurations

| Model Type | Config File | Description |
|------------|-------------|-------------|
| 2-Stage Hierarchical | `training_2stage_tinystories.json` | Full hierarchical with dynamic chunking |
| 1-Stage Hierarchical | `training_1stage_tinystories.json` | Single hierarchy level |
| Flat Baseline | `training_flat_tinystories.json` | Standard flat model for comparison |

## References

- [MatMul-Free LM](https://github.com/ridgerchu/matmulfreellm) - Original BitLinear and HGRN implementation
- [HGRN2 Paper](https://arxiv.org/abs/2404.07904) - Gated Linear RNNs with State Expansion
- [BitNet Paper](https://arxiv.org/abs/2310.11453) - Scaling 1-bit Transformers

## License

MIT License
