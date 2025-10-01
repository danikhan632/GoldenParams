#!/usr/bin/env python3
"""
Test that MaskedAdamW handles shape mismatches gracefully.
"""

import torch
import torch.nn as nn
from golden_params.optimizers.masked_adamw import MaskedAdamW
from golden_params.utils import convert_masks_to_torch_sparse_coo

def test_shape_mismatch():
    """Test that mismatched shapes are handled gracefully."""
    print("Testing shape mismatch handling...")
    print("=" * 60)

    # Create a model
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.Linear(5, 1)
    )

    # Create masks with WRONG shapes (simulating loading masks from different model)
    wrong_masks = {
        "0.weight": torch.rand(100, 200) > 0.5,  # Wrong shape!
        "0.bias": torch.rand(5) > 0.5,           # Correct shape
        "1.weight": torch.rand(50, 50) > 0.5,    # Wrong shape!
    }

    print("Model parameter shapes:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}")

    print("\nMask shapes:")
    for name, mask in wrong_masks.items():
        print(f"  {name}: {mask.shape}")

    # Convert to sparse
    sparse_masks = convert_masks_to_torch_sparse_coo(wrong_masks)

    # Create optimizer (should not crash)
    print("\nCreating MaskedAdamW optimizer...")
    optimizer = MaskedAdamW(
        model.parameters(),
        sparse_masks=sparse_masks,
        masked_lr=1e-3,
        unmasked_lr=1e-5
    )

    # Add parameter names
    optimizer.add_param_names(dict(model.named_parameters()))

    print("✓ Optimizer created successfully")

    # Run a training step (should work despite shape mismatches)
    print("\nRunning training step...")
    x = torch.randn(8, 10)
    y = torch.randn(8, 1)
    loss = nn.MSELoss()(model(x), y)
    loss.backward()

    optimizer.step()
    print("✓ Training step completed successfully")

    # Check optimizer info
    lr_info = optimizer.get_lr_info()
    print("\nOptimizer info:")
    print(f"  Masked LR: {lr_info['masked_lr']}")
    print(f"  Unmasked LR: {lr_info['unmasked_lr']}")
    print(f"  Masks loaded: {lr_info['masks_loaded']}")
    print(f"  Parameters mapped: {lr_info['param_name_mapping']}")
    print(f"  Skipped masks: {lr_info['skipped_masks']}")
    print(f"  Skipped mask names: {lr_info['skipped_mask_names']}")

    # Verify that mismatched masks were skipped
    assert lr_info['skipped_masks'] == 2, \
        f"Expected 2 skipped masks, got {lr_info['skipped_masks']}"
    assert "0.weight" in lr_info['skipped_mask_names'], \
        "0.weight should be in skipped masks"
    assert "1.weight" in lr_info['skipped_mask_names'], \
        "1.weight should be in skipped masks"

    print("\n" + "=" * 60)
    print("🎉 Shape mismatch handling test passed!")
    print("=" * 60)

if __name__ == "__main__":
    test_shape_mismatch()
