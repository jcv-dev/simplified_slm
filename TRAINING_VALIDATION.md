# Training Implementation Validation

## Comparison with Reference Implementations

This document compares the training infrastructure in `simplified_slm` against the reference implementations from `hnet-main` and `matmulfreellm-master`.

---

## ✅ What We Got Right

### 1. **Optimizer Configuration** ✓
Our implementation correctly follows standard practices:
- **AdamW** with weight decay (like both reference repos)
- **Separate parameter groups** for decay/no-decay (bias, norms excluded from decay)
- **Standard hyperparameters**: β₁=0.9, β₂=0.95, ε=1e-8

```python
# simplified_slm/training/optimizer.py
def build_optimizer(model_parameters, config):
    decay_params = []
    no_decay_params = []
    
    for name, param in model_parameters:
        if param.dim() < 2 or 'bias' in name or 'norm' in name:
            no_decay_params.append(param)  # ✓ Correct
        else:
            decay_params.append(param)
```

**Reference (hnet-main):**
```python
# hnet/utils/train.py
if name.endswith(".bias") or ".norm." in name:
    apply_optimization_params(param, weight_decay=0.0)  # Same pattern
```

---

### 2. **Learning Rate Scheduling** ✓
- **Cosine annealing** with linear warmup (standard)
- **Linear option** also available
- **Warmup steps** to prevent early instability

```python
# simplified_slm/training/optimizer.py
def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))  # ✓ Linear warmup
    
    progress = (current_step - warmup_steps) / (max_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))  # ✓ Cosine decay
```

---

### 3. **Mixed Precision Training** ✓
- **BF16** support (preferred for modern GPUs)
- **FP16** support with gradient scaling
- **Automatic Mixed Precision** (AMP) using `torch.amp.autocast`

```python
# simplified_slm/training/trainer.py
with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
    outputs = self.model(input_ids=batch['input_ids'], labels=batch['labels'])
    loss = outputs.loss / self.config.gradient_accumulation_steps
```

✓ This matches MatMulFreeLM's approach (they use `.half()` which is FP16)

---

### 4. **Gradient Accumulation** ✓
- Correctly divides loss by accumulation steps
- Only steps optimizer after accumulation completes
- Proper gradient clipping before optimizer step

```python
loss = outputs.loss / self.config.gradient_accumulation_steps
self.scaler.scale(loss).backward()
accumulation_loss += loss.item()

if micro_step % self.config.gradient_accumulation_steps == 0:
    self.scaler.unscale_(self.optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
    self.scaler.step(self.optimizer)
    self.scheduler.step()
```

✓ Standard implementation, no issues

---

### 5. **Checkpointing** ✓
- Saves model state, optimizer state, scheduler state
- Includes training config and model config
- Can resume from checkpoints
- Tracks best validation loss

```python
torch.save({
    'model_state_dict': self.model.state_dict(),
    'optimizer_state_dict': self.optimizer.state_dict(),
    'scheduler_state_dict': self.scheduler.state_dict(),
    'global_step': self.global_step,
    'config': config_dict,
}, path)
```

✓ Complete checkpoint implementation

---

### 6. **Byte-Level Tokenization** ✓
- Fixed vocabulary size of 256 ✓
- Direct UTF-8 encoding ✓
- No tokenizer training needed ✓
- Compatible with HNetBit models ✓

```python
# simplified_slm/utils/tokenizers.py
class ByteTokenizer:
    vocab_size = 256  # All possible byte values
    bos_idx = 254
    eos_idx = 255
    pad_idx = 0
```

✓ Matches H-Net's ByteTokenizer perfectly

---

### 7. **Dataset Loading** ✓
- HuggingFace datasets integration ✓
- Proper byte-level conversion ✓
- Caching support ✓
- Train/val splits ✓

---

## ⚠️ Missing Features from H-Net

### 1. **Per-Stage Learning Rate Multipliers** ⚠️

**H-Net uses stage-specific learning rates for hierarchical models:**

```python
# hnet/models/hnet.py
def _apply_lr_multiplier(self, lr_multiplier: list[float]) -> None:
    """Applies learning rate multipliers per stage."""
    for param in self.parameters():
        apply_optimization_params(param, lr_multiplier=lr_multiplier[self.stage_idx])
    
    if not self.is_innermost:
        self.main_network._apply_lr_multiplier(lr_multiplier)
```

**Why this matters:**
- Different hierarchy stages learn at different rates
- Outer stages (coarse-grained) often need higher LR
- Inner stages (fine-grained) often need lower LR
- Example: `[3.0, 1.7, 0.9]` for 3 stages

**Our implementation:** Uses same LR for all parameters ❌

**Recommendation:** Add support for `lr_multiplier` in HNetBit training:

```python
# Proposed addition to simplified_slm/training/optimizer.py
def build_optimizer_hierarchical(model, config):
    """Build optimizer with per-stage learning rates for HNetBit."""
    lr_multipliers = config.get('lr_multipliers', None)
    
    if lr_multipliers and hasattr(model, '_apply_lr_multiplier'):
        model._apply_lr_multiplier(lr_multipliers)
        
        # Group parameters by their _optim attribute
        param_groups = []
        seen_configs = {}
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            optim_config = getattr(param, '_optim', {})
            lr_mult = optim_config.get('lr_multiplier', 1.0)
            wd = optim_config.get('weight_decay', config.weight_decay)
            
            # Group by (lr_mult, weight_decay)
            key = (lr_mult, wd)
            if key not in seen_configs:
                seen_configs[key] = []
            seen_configs[key].append(param)
        
        for (lr_mult, wd), params in seen_configs.items():
            param_groups.append({
                'params': params,
                'lr': config.learning_rate * lr_mult,
                'weight_decay': wd,
            })
        
        return AdamW(param_groups, betas=(config.adam_beta1, config.adam_beta2), eps=config.adam_eps)
    
    # Fallback to standard optimizer
    return build_optimizer(model.named_parameters(), config)
```

---

### 2. **Load Balancing Loss** ⚠️

**H-Net includes auxiliary loss for dynamic chunking:**

```python
# hnet/utils/train.py
def load_balancing_loss(router_output, N):
    """Compute load balancing loss for routing module."""
    boundary_prob = router_output.boundary_prob
    tokenized_prob = boundary_prob[..., -1]
    boundary_mask = router_output.boundary_mask
    
    true_ratio = boundary_mask.float().mean()
    average_prob = tokenized_prob.float().mean()
    
    return (
        (1 - true_ratio) * (1 - average_prob) +
        (true_ratio) * (average_prob) * (N-1)
    ) * N / (N-1)
```

**Why this matters:**
- Encourages balanced chunking across sequences
- Prevents degenerate solutions (all chunks or no chunks)
- Critical for hierarchical models with dynamic chunking

**Our implementation:** Standard cross-entropy only ❌

**Recommendation:** Add load balancing loss to HNetBit training:

```python
# Proposed addition to simplified_slm/training/trainer.py
def compute_loss_with_load_balancing(self, batch, lambda_lb=0.01):
    """Compute total loss including load balancing."""
    outputs = self.model(
        input_ids=batch['input_ids'],
        attention_mask=batch.get('attention_mask'),
        labels=batch['labels'],
    )
    
    ce_loss = outputs.loss
    
    # Add load balancing loss if model returns routing outputs
    if hasattr(outputs, 'router_outputs') and outputs.router_outputs:
        lb_loss = 0.0
        num_stages = len(self.model.config.d_model)
        
        for router_output in outputs.router_outputs:
            if router_output is not None:
                # Compute load balancing for this stage
                boundary_prob = router_output.boundary_prob
                tokenized_prob = boundary_prob[..., -1]
                boundary_mask = router_output.boundary_mask
                
                true_ratio = boundary_mask.float().mean()
                average_prob = tokenized_prob.float().mean()
                
                # Downsampling factor (e.g., 2-3x expected)
                N = 2.5  # Can be computed from config
                
                stage_lb_loss = (
                    (1 - true_ratio) * (1 - average_prob) +
                    (true_ratio) * (average_prob) * (N-1)
                ) * N / (N-1)
                
                lb_loss += stage_lb_loss
        
        total_loss = ce_loss + lambda_lb * lb_loss
        return total_loss, ce_loss.item(), lb_loss.item()
    
    return ce_loss, ce_loss.item(), 0.0
```

---

### 3. **Custom Parameter Grouping Pattern** ℹ️

**H-Net's approach:**
```python
# Annotate parameters with custom optimization settings
for name, param in model.named_parameters():
    if name.endswith(".bias") or ".norm." in name:
        apply_optimization_params(param, weight_decay=0.0)

# Later, group by _optim attribute
param_groups = []
for name, param in model.named_parameters():
    current_tuple = tuple(param._optim.get(key, None) for key in all_keys)
    # ... group parameters with same settings
```

**Our approach:** Direct parameter group creation in optimizer builder

**Assessment:** Both approaches are valid. H-Net's is more flexible for complex models, ours is simpler and more standard. ✓ Fine as-is for current needs.

---

## 📊 MatMulFreeLM Comparison

**Key Finding:** MatMulFreeLM repository **does not include training code**. They:
- Provide pre-trained models via HuggingFace
- Use standard HuggingFace Trainer (implied)
- Don't expose custom training logic

**Their models use:**
- Standard tokenizers (not byte-level) - different from our approach
- FusedBitLinear layers (same concept as our BitLinear)
- HGRN attention (same as our implementation)

**Validation:** Our BitLinear and HGRN implementations are compatible ✓

---

## 🎯 Recommendations

### Priority 1: Add Load Balancing Loss ⚠️
For HNetBit 2-stage models, this is **critical** for proper training:

1. Modify `HNetBitForCausalLM` to return router outputs in `CausalLMOutputWithPast`
2. Add `load_balancing_loss` function to `training/trainer.py`
3. Add `lambda_lb` hyperparameter to training configs (recommended: 0.01-0.1)

### Priority 2: Add Per-Stage LR Multipliers ⚠️
For HNetBit models, this improves training stability:

1. Add `_apply_lr_multiplier` method to `HNetBit` model
2. Extend `build_optimizer` to handle `lr_multipliers` config
3. Add `lr_multipliers: [2.0, 1.5, 1.0]` to training configs

### Priority 3: Enhanced Monitoring ✓ (Already Good)
Current implementation is solid:
- TensorBoard integration ✓
- Ternary weight statistics ✓
- Gradient norms ✓
- Generation samples ✓

---

## ✅ Current Implementation: Production Ready For Flat Models

**For SimplifiedSLM (flat model):** ✓ Training infrastructure is complete and correct
- All standard components properly implemented
- Mixed precision support
- Dataset loading and evaluation
- Comprehensive logging

**For HNetBit (hierarchical model):** ⚠️ Missing H-Net specific features
- Needs load balancing loss for dynamic chunking
- Would benefit from per-stage learning rate multipliers
- Otherwise core training loop is solid

---

## Summary Table

| Feature | SimplifiedSLM | HNetBit | H-Net Ref | MatMulFree Ref |
|---------|---------------|---------|-----------|----------------|
| Basic Training Loop | ✓ | ✓ | ✓ | N/A |
| AdamW Optimizer | ✓ | ✓ | ✓ | ✓ (implied) |
| LR Scheduling | ✓ | ✓ | ✓ | ✓ (implied) |
| Mixed Precision | ✓ | ✓ | ✓ | ✓ |
| Gradient Clipping | ✓ | ✓ | ✓ | ✓ (implied) |
| Checkpointing | ✓ | ✓ | ✓ | ✓ |
| Per-Stage LR | N/A | ❌ | ✓ | N/A |
| Load Balancing Loss | N/A | ❌ | ✓ | N/A |
| Byte Tokenization | ✓ | ✓ | ✓ | ❌ (standard) |

**Legend:** ✓ = Implemented, ❌ = Missing, N/A = Not applicable

---

## Quick Fix Checklist

- [ ] Add load balancing loss computation function
- [ ] Modify HNetBitForCausalLM to return router outputs
- [ ] Add `_apply_lr_multiplier` to HNetBit model
- [ ] Extend optimizer builder for hierarchical LR
- [ ] Add `lr_multipliers` and `lambda_lb` to config
- [ ] Test on small HNetBit model
- [ ] Document hierarchical training nuances

---

## Conclusion

**Your training implementation is solid** for flat models and follows best practices from both reference repositories. The two missing features (per-stage LR and load balancing loss) are **specific to H-Net's hierarchical architecture** and should be added for optimal HNetBit training.

The core infrastructure (optimizer, scheduler, mixed precision, data loading, evaluation) is all implemented correctly and ready for production use.
