#!/usr/bin/env python3
"""
Test script for MaskedAdamW optimizer with sparse COO tensor masks.
"""

import torch
import torch.nn as nn
from golden_params.optimizers.masked_adamw import MaskedAdamW
from golden_params.utils import convert_masks_to_sparse_coo
from golden_params.sparse_mask_utils import SparseMaskManager


def test_basic_functionality():
    """Test basic MaskedAdamW functionality."""
    print("Testing basic MaskedAdamW functionality...")

    # Create a simple model
    model = nn.Sequential(
        nn.Linear(4, 3),
        nn.Linear(3, 1)
    )

    # Create masks - 50% sparse for demonstration
    masks = {
        "0.weight": torch.rand(3, 4) > 0.5,  # First layer weight
        "0.bias": torch.rand(3) > 0.5,       # First layer bias
        "1.weight": torch.rand(1, 3) > 0.5,  # Second layer weight
        "1.bias": torch.rand(1) > 0.5,       # Second layer bias
    }

    print(f"Created masks for {len(masks)} parameters")
    for name, mask in masks.items():
        density = mask.sum().item() / mask.numel()
        print(f"  {name}: shape {mask.shape}, density {density:.1%}")

    # Convert to sparse COO
    sparse_masks = convert_masks_to_sparse_coo(masks)

    # Create optimizer
    optimizer = MaskedAdamW(
        model.parameters(),
        sparse_masks=sparse_masks,
        masked_lr=1e-2,      # Higher learning rate for masked parameters
        unmasked_lr=1e-4,    # Lower learning rate for unmasked parameters
        weight_decay=1e-3
    )

    # Add parameter name mapping
    param_names = {}
    for name, param in model.named_parameters():
        param_names[name] = param
    optimizer.add_param_names(param_names)

    # Test optimization step
    x = torch.randn(10, 4)
    y = torch.randn(10, 1)

    # Forward pass
    output = model(x)
    loss = nn.MSELoss()(output, y)

    print(f"Initial loss: {loss.item():.6f}")

    # Backward pass
    loss.backward()

    # Check gradients exist
    total_grad_norm = 0
    param_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm
            param_count += 1
            print(f"  {name}: grad norm = {grad_norm:.6f}")

    print(f"Total gradient norm: {total_grad_norm:.6f} across {param_count} parameters")

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    # Check optimizer info
    lr_info = optimizer.get_lr_info()
    print(f"LR Info: {lr_info}")

    print("✅ Basic functionality test passed!")
    return optimizer


def test_learning_rate_application():
    """Test that different learning rates are actually applied."""
    print("\nTesting learning rate application...")

    # Create a simple 1-parameter model for easy testing
    param = nn.Parameter(torch.ones(2, 2))

    # Create mask where only top-left element is masked (True)
    mask = torch.tensor([[True, False], [False, False]])
    sparse_mask = convert_masks_to_sparse_coo({"param": mask})

    # Create optimizer with very different learning rates
    optimizer = MaskedAdamW(
        [param],
        sparse_masks=sparse_mask,
        masked_lr=1.0,    # Very high LR for masked
        unmasked_lr=0.1,  # Lower LR for unmasked
        weight_decay=0.0  # No weight decay for cleaner test
    )

    optimizer.add_param_names({"param": param})

    print(f"Initial parameter:\n{param}")
    print(f"Mask pattern:\n{mask}")

    # Create gradient
    param.grad = torch.ones_like(param)
    print(f"Gradient:\n{param.grad}")

    # Take optimization step
    param_before = param.data.clone()
    optimizer.step()
    param_after = param.data.clone()

    update = param_before - param_after
    print(f"Parameter update:\n{update}")

    # Check that masked position (0,0) has larger update than unmasked positions
    masked_update = abs(update[0, 0].item())
    unmasked_update = abs(update[0, 1].item())  # Any unmasked position

    print(f"Masked position update magnitude: {masked_update:.6f}")
    print(f"Unmasked position update magnitude: {unmasked_update:.6f}")

    # The masked position should have a larger update due to higher learning rate
    assert masked_update > unmasked_update, \
        f"Masked update ({masked_update}) should be larger than unmasked update ({unmasked_update})"

    print("✅ Learning rate application test passed!")


def test_sparse_efficiency():
    """Test memory efficiency with sparse masks."""
    print("\nTesting sparse mask efficiency...")

    # Create a large parameter with very sparse mask (1% density)
    large_param = nn.Parameter(torch.randn(100, 100))
    sparse_mask_dense = torch.rand(100, 100) > 0.99  # 1% density

    print(f"Large parameter shape: {large_param.shape}")
    print(f"Mask density: {sparse_mask_dense.sum().item()}/{sparse_mask_dense.numel()} = "
          f"{sparse_mask_dense.sum().item()/sparse_mask_dense.numel():.1%}")

    # Convert to sparse
    sparse_masks = convert_masks_to_sparse_coo({"large_param": sparse_mask_dense})

    # Check memory usage
    sparse_mask = sparse_masks["large_param"]
    memory_info = SparseMaskManager.get_sparse_memory_usage(sparse_mask)

    print(f"Memory usage comparison:")
    print(f"  Dense mask: {memory_info['dense_memory_bytes']} bytes")
    print(f"  Sparse mask: {memory_info['sparse_memory_bytes']} bytes")
    print(f"  Savings: {memory_info['memory_savings_bytes']} bytes "
          f"({memory_info['memory_savings_ratio']:.1%})")

    # Create optimizer
    optimizer = MaskedAdamW(
        [large_param],
        sparse_masks=sparse_masks,
        masked_lr=1e-3,
        unmasked_lr=1e-4
    )

    optimizer.add_param_names({"large_param": large_param})

    # Test a training step
    large_param.grad = torch.randn_like(large_param) * 0.01

    try:
        optimizer.step()
        print("✅ Sparse efficiency test passed!")
    except Exception as e:
        print(f"❌ Sparse efficiency test failed: {e}")
        raise


def test_no_mask_fallback():
    """Test that optimizer works correctly when no mask is provided for a parameter."""
    print("\nTesting no-mask fallback behavior...")

    model = nn.Linear(3, 2)

    # Only create mask for weight, not bias
    masks = {
        "weight": torch.rand(2, 3) > 0.5
    }
    sparse_masks = convert_masks_to_sparse_coo(masks)

    optimizer = MaskedAdamW(
        model.parameters(),
        sparse_masks=sparse_masks,
        masked_lr=1e-2,
        unmasked_lr=1e-4
    )

    # Map parameter names
    param_names = dict(model.named_parameters())
    optimizer.add_param_names(param_names)

    # Create some gradients
    x = torch.randn(5, 3)
    y = torch.randn(5, 2)
    loss = nn.MSELoss()(model(x), y)
    loss.backward()

    print("Parameters and their gradients:")
    for name, param in model.named_parameters():
        has_mask = name in sparse_masks
        grad_norm = param.grad.norm().item() if param.grad is not None else 0.0
        print(f"  {name}: has_mask={has_mask}, grad_norm={grad_norm:.6f}")

    # This should work without error
    try:
        optimizer.step()
        print("✅ No-mask fallback test passed!")
    except Exception as e:
        print(f"❌ No-mask fallback test failed: {e}")
        raise


def run_all_tests():
    """Run all test cases."""
    print("=" * 60)
    print("Testing MaskedAdamW Optimizer")
    print("=" * 60)

    test_basic_functionality()
    test_learning_rate_application()
    test_sparse_efficiency()
    test_no_mask_fallback()

    print("\n" + "=" * 60)
    print("🎉 All MaskedAdamW tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()