# MaskedAdamW Optimizer - Usage Guide

## Overview

The `MaskedAdamW` optimizer extends PyTorch's AdamW to support **dual learning rates** based on sparse COO tensor masks. This allows you to apply different learning rates to different parameter regions efficiently.

## Key Features

- **Dual Learning Rates**: Apply `masked_lr` to parameters marked True in the mask, and `unmasked_lr` to parameters marked False
- **Sparse COO Integration**: Uses PyTorch's sparse COO tensors for memory-efficient mask storage (up to 75% memory savings)
- **Single Optimizer Instance**: No need to maintain separate optimizers or duplicate state
- **Efficient Updates**: Splits gradients into masked/unmasked components and applies appropriate learning rates

## Installation

The optimizer is part of the `golden_params` package:

```python
from golden_params.optimizers.masked_adamw import MaskedAdamW
from golden_params.utils import convert_masks_to_torch_sparse_coo
```

## Basic Usage

### 1. Load or Create Masks

If you have dense boolean masks from the golden params export:

```python
import torch

# Load masks from file
mask_data = torch.load("golden_masks.pt", map_location='cpu')
masks = mask_data.get("masks", {})  # Dict[str, torch.BoolTensor]
```

### 2. Convert to Sparse COO Format

```python
from golden_params.utils import convert_masks_to_torch_sparse_coo

# Convert dense boolean masks to sparse COO tensors
sparse_masks = convert_masks_to_torch_sparse_coo(masks)
```

### 3. Create Optimizer

```python
from golden_params.optimizers.masked_adamw import MaskedAdamW

optimizer = MaskedAdamW(
    model.parameters(),
    sparse_masks=sparse_masks,
    masked_lr=1e-3,      # Learning rate for masked parameters (higher)
    unmasked_lr=1e-5,    # Learning rate for unmasked parameters (lower)
    weight_decay=1e-2,
    betas=(0.9, 0.999),
    eps=1e-8
)
```

### 4. Map Parameter Names

The optimizer needs to know which parameters correspond to which masks:

```python
# Add parameter name mapping
param_names = dict(model.named_parameters())
optimizer.add_param_names(param_names)
```

### 5. Use Like Normal AdamW

```python
# Standard training loop
for batch in dataloader:
    # Forward pass
    loss = model(batch)

    # Backward pass
    loss.backward()

    # Optimizer step (applies dual learning rates automatically)
    optimizer.step()
    optimizer.zero_grad()
```

## Complete Example

```python
import torch
from torch import nn
from golden_params.optimizers.masked_adamw import MaskedAdamW
from golden_params.utils import convert_masks_to_torch_sparse_coo

# 1. Load your model
model = nn.Sequential(
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

# 2. Load and convert masks
mask_data = torch.load("golden_masks.pt", map_location='cpu')
dense_masks = mask_data["masks"]
sparse_masks = convert_masks_to_torch_sparse_coo(dense_masks)

print(f"Loaded {len(sparse_masks)} sparse masks")

# 3. Create optimizer with dual learning rates
optimizer = MaskedAdamW(
    model.parameters(),
    sparse_masks=sparse_masks,
    masked_lr=1e-3,    # 100x higher for important parameters
    unmasked_lr=1e-5,  # Lower for less important parameters
    weight_decay=0.01
)

# 4. Add parameter names
optimizer.add_param_names(dict(model.named_parameters()))

# 5. Get optimizer info
lr_info = optimizer.get_lr_info()
print(f"Masked LR: {lr_info['masked_lr']}")
print(f"Unmasked LR: {lr_info['unmasked_lr']}")
print(f"Total masked parameters: {lr_info['total_masked_parameters']:,}")
print(f"Total unmasked parameters: {lr_info['total_unmasked_parameters']:,}")

# 6. Train normally
for epoch in range(10):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch).sum()
        loss.backward()
        optimizer.step()
```

## GRPO Integration Example

```python
# In your GRPO training script
from trl import GRPOConfig, GRPOTrainer
from golden_params.optimizers.masked_adamw import MaskedAdamW
from golden_params.utils import convert_masks_to_torch_sparse_coo

# Load masks
mask_data = torch.load(args.golden_mask_path, map_location='cpu')
masks = mask_data.get("masks", {})
sparse_masks = convert_masks_to_torch_sparse_coo(masks)

# Create optimizer function
def create_optimizer(model):
    optimizer = MaskedAdamW(
        model.parameters(),
        sparse_masks=sparse_masks,
        masked_lr=args.masked_lr,
        unmasked_lr=args.unmasked_lr,
        weight_decay=0.01
    )
    optimizer.add_param_names(dict(model.named_parameters()))
    return optimizer

# Override trainer's optimizer
trainer = GRPOTrainer(...)
trainer.optimizer = create_optimizer(trainer.model)
```

## Memory Efficiency

The sparse COO format provides significant memory savings for sparse masks:

```python
from golden_params.sparse_mask_utils import SparseMaskManager

# Check memory usage
for name, mask in sparse_masks.items():
    memory_info = SparseMaskManager.get_sparse_memory_usage(mask)
    print(f"{name}:")
    print(f"  Sparse: {memory_info['sparse_memory_bytes']:,} bytes")
    print(f"  Dense: {memory_info['dense_memory_bytes']:,} bytes")
    print(f"  Savings: {memory_info['memory_savings_ratio']:.1%}")
```

For a 5% sparse mask, you typically see **~75% memory savings**.

## Fallback Behavior

If a parameter doesn't have a corresponding mask, it will use the `unmasked_lr`:

```python
# These masks don't cover all model parameters
sparse_masks = {
    "layer1.weight": torch.sparse_coo_tensor(...),
    # layer1.bias is missing
}

# This works - layer1.bias will use unmasked_lr
optimizer = MaskedAdamW(model.parameters(), sparse_masks, ...)
```

## Troubleshooting

### "Mask shape does not match parameter shape"

Ensure your mask shapes exactly match your parameter shapes:

```python
for name, param in model.named_parameters():
    if name in sparse_masks:
        mask = sparse_masks[name]
        print(f"{name}: param {param.shape}, mask {mask.shape}")
        assert param.shape == mask.shape
```

### "'dict' object has no attribute 'is_sparse'"

Make sure you're using `convert_masks_to_torch_sparse_coo`, not the older `convert_masks_to_sparse_coo`:

```python
# ✓ Correct
from golden_params.utils import convert_masks_to_torch_sparse_coo
sparse_masks = convert_masks_to_torch_sparse_coo(masks)

# ✗ Wrong (old function for different format)
from golden_params.utils import convert_masks_to_sparse_coo
```

### No parameter names provided warning

Add parameter names to enable mask application:

```python
optimizer.add_param_names(dict(model.named_parameters()))
```

## Performance Tips

1. **Use appropriate learning rate ratios**: A common ratio is 10x-100x higher for masked parameters
2. **Check mask density**: Very sparse masks (< 1%) provide the most memory benefit
3. **Verify mask coverage**: Use `get_lr_info()` to check how many parameters are masked
4. **Profile memory usage**: Use `SparseMaskManager.get_sparse_memory_usage()` to verify savings

## API Reference

### MaskedAdamW Constructor

```python
MaskedAdamW(
    params,                    # Model parameters (iterable)
    sparse_masks: Dict[str, torch.Tensor],  # Parameter name -> sparse COO mask
    masked_lr: float = 1e-3,  # LR for masked (True) parameters
    unmasked_lr: float = 1e-4,  # LR for unmasked (False) parameters
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
    amsgrad: bool = False,
    maximize: bool = False,
    foreach: Optional[bool] = None,
    capturable: bool = False,
    differentiable: bool = False,
    fused: Optional[bool] = None
)
```

### Key Methods

- `add_param_names(param_names: Dict[str, Parameter])`: Map parameter names to tensors
- `get_lr_info()`: Get learning rate and mask statistics
- `step(closure=None)`: Perform optimization step with dual learning rates
- `zero_grad()`: Clear gradients (inherited from AdamW)

## Testing

Run the test suite to verify installation:

```bash
python3 test_mask_loading.py
```

Expected output:
```
✓ Conversion successful!
✓ All masks are sparse after conversion: True
✓ Optimizer step completed successfully
🎉 All tests passed!
```
