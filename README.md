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
│   ├── config.py       # SimplifiedSLMConfig
│   └── modeling.py     # SimplifiedSLMForCausalLM
├── layers/
│   └── hgrn_bit.py     # HGRNBitAttention, HGRNBitMLP, HGRNBitBlock
├── ops/
│   ├── bitnet.py       # BitLinear, weight_quant, activation_quant
│   ├── activations.py  # SwiGLU, swish
│   ├── fused_norm_gate.py  # FusedRMSNormSwishGate
│   └── hgrn/
│       ├── chunk.py        # Chunk-parallel HGRN
│       └── recurrent_fuse.py  # Fused recurrent HGRN
├── utils/
│   ├── tokenizers.py   # ByteTokenizer
│   ├── cache.py        # RecurrentCache
│   └── helpers.py      # Utility functions
├── configs/
│   └── slm_base.json   # Default configuration
├── tests/
│   └── test_model.py   # Unit tests
└── generate.py         # Text generation script
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
python generate.py --prompt "Hello" --max_tokens 100 --temperature 0.8
python generate.py --model_path ./checkpoint.pt --prompt "Once upon a time" --greedy
```

## Training (Coming in Objective 2)

Training scripts and dataset preparation will be added in Objective 2.

## References

- [MatMul-Free LM](https://github.com/ridgerchu/matmulfreellm) - Original BitLinear and HGRN implementation
- [HGRN2 Paper](https://arxiv.org/abs/2404.07904) - Gated Linear RNNs with State Expansion
- [BitNet Paper](https://arxiv.org/abs/2310.11453) - Scaling 1-bit Transformers

## License

MIT License
