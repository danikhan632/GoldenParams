#!/usr/bin/env python3
"""
Test script to verify golden mask loading and conversion works correctly.
"""

import torch
from golden_params.utils import convert_masks_to_torch_sparse_coo
from golden_params.optimizers.masked_adamw import MaskedAdamW

def test_mask_loading():
    """Test loading and converting masks from the golden_masks.pt file."""
    print("Testing mask loading and conversion...")
    print("=" * 60)

    # Load the mask file
    mask_path = "/home/green/code/golden/examples/golden_masks.pt"
    print(f"Loading masks from: {mask_path}")

    try:
        mask_data = torch.load(mask_path, map_location='cpu')
        masks = mask_data.get("masks", mask_data.get("intersection_masks", {}))

        print(f"✓ Loaded {len(masks)} masks")

        # Check if conversion is needed
        all_sparse = all(
            isinstance(m, torch.Tensor) and m.is_sparse
            for m in masks.values()
        )

        print(f"  All masks are sparse: {all_sparse}")

        if not all_sparse:
            print("  Converting dense boolean masks to sparse COO tensors...")
            sparse_masks = convert_masks_to_torch_sparse_coo(masks)
            print(f"✓ Conversion successful!")
        else:
            sparse_masks = masks
            print("  No conversion needed")

        # Verify all are sparse now
        all_sparse_after = all(m.is_sparse for m in sparse_masks.values())
        print(f"  All masks are sparse after conversion: {all_sparse_after}")

        # Sample a few masks
        print("\nSample mask verification (first 3):")
        for i, (name, mask) in enumerate(list(sparse_masks.items())[:3]):
            print(f"\n  {i+1}. {name}")
            print(f"     Shape: {mask.shape}")
            print(f"     Is sparse: {mask.is_sparse}")
            print(f"     Sparse nnz: {mask._nnz()}")

            # Verify we can convert back to dense
            dense = mask.to_dense()
            density = (dense != 0).sum().item() / dense.numel()
            print(f"     Density: {density:.2%}")

            # Memory savings
            sparse_mem = (mask.indices().element_size() * mask.indices().numel() +
                         mask.values().element_size() * mask.values().numel())
            dense_mem = dense.numel() * dense.element_size()
            savings = (dense_mem - sparse_mem) / dense_mem * 100
            print(f"     Memory savings: {savings:.1f}%")

        # Test with a small model
        print("\n" + "=" * 60)
        print("Testing MaskedAdamW with converted masks...")

        # Create a small dummy model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.Linear(5, 1)
        )

        # Create test masks with correct shapes for our dummy model
        # Model has Linear(10, 5) and Linear(5, 1), so:
        # - "0.weight" should be shape (5, 10)
        # - "0.bias" should be shape (5,)
        test_masks_dense = {
            "0.weight": torch.rand(5, 10) > 0.5,  # Random 50% mask
            "0.bias": torch.rand(5) > 0.5  # Random 50% mask
        }
        test_masks = convert_masks_to_torch_sparse_coo(test_masks_dense)

        print(f"  Using {len(test_masks)} test masks")

        # Create optimizer
        optimizer = MaskedAdamW(
            model.parameters(),
            sparse_masks=test_masks,
            masked_lr=1e-3,
            unmasked_lr=1e-5
        )

        # Add parameter names
        optimizer.add_param_names(dict(model.named_parameters()))

        print("✓ MaskedAdamW optimizer created successfully")

        # Test a training step
        x = torch.randn(8, 10)
        y = torch.randn(8, 1)
        loss = torch.nn.MSELoss()(model(x), y)
        loss.backward()

        print("  Running optimizer step...")
        optimizer.step()
        print("✓ Optimizer step completed successfully")

        # Get optimizer info
        lr_info = optimizer.get_lr_info()
        print(f"\nOptimizer info:")
        print(f"  Masked LR: {lr_info['masked_lr']}")
        print(f"  Unmasked LR: {lr_info['unmasked_lr']}")
        print(f"  Masks loaded: {lr_info['masks_loaded']}")

        print("\n" + "=" * 60)
        print("🎉 All tests passed!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = test_mask_loading()
    exit(0 if success else 1)
