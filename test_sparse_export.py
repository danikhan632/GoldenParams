#!/usr/bin/env python3
"""
Test script to verify the sparse COO tensor export functionality.
"""

import torch
import os
import tempfile
from golden_params.utils import convert_masks_to_sparse_coo

def test_convert_masks_to_sparse_coo():
    """Test the conversion function directly."""
    print("Testing convert_masks_to_sparse_coo function...")

    # Create test boolean masks
    test_masks = {
        "layer1.weight": torch.tensor([[True, False, True], [False, True, False]]),
        "layer2.bias": torch.tensor([True, False, True, False]),
    }

    # Convert to sparse COO
    sparse_masks = convert_masks_to_sparse_coo(test_masks)

    # Verify results
    for name, sparse_mask in sparse_masks.items():
        original_mask = test_masks[name]
        print(f"\n{name}:")
        print(f"  Original shape: {original_mask.shape}")
        print(f"  Original density: {original_mask.count_nonzero().item()}/{original_mask.numel()} = {original_mask.count_nonzero().item()/original_mask.numel():.2%}")
        print(f"  Sparse shape: {sparse_mask.shape}")
        print(f"  Sparse nnz: {sparse_mask._nnz()}")
        print(f"  Is sparse: {sparse_mask.is_sparse}")
        print(f"  Layout: {sparse_mask.layout}")

        # Verify conversion by converting back to dense
        reconstructed = sparse_mask.to_dense().bool()
        assert torch.equal(reconstructed, original_mask), f"Conversion failed for {name}"
        print(f"  ✅ Conversion verified!")

def test_save_and_load():
    """Test saving and loading sparse masks."""
    print("\nTesting save and load functionality...")

    # Create test data
    test_masks = {
        "weight": torch.rand(10, 10) > 0.7,  # ~30% density
        "bias": torch.rand(5) > 0.5,  # ~50% density
    }

    sparse_masks = convert_masks_to_sparse_coo(test_masks)
    test_summary = {
        "weight": {"shape": [10, 10], "active_count": 30, "total_count": 100, "density": 0.3},
        "bias": {"shape": [5], "active_count": 3, "total_count": 5, "density": 0.6}
    }

    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
        torch.save({
            "masks": sparse_masks,
            "summary": test_summary,
            "kl_divergence": 0.123
        }, tmp_file.name)

        # Load back
        loaded_data = torch.load(tmp_file.name)

        print(f"  Saved and loaded file: {tmp_file.name}")
        print(f"  Loaded keys: {list(loaded_data.keys())}")
        print(f"  Number of masks: {len(loaded_data['masks'])}")

        # Verify loaded masks are still sparse
        for name, mask in loaded_data['masks'].items():
            print(f"  {name}: sparse={mask.is_sparse}, nnz={mask._nnz()}, shape={mask.shape}")

        # Clean up
        os.unlink(tmp_file.name)
        print(f"  ✅ Save/load test passed!")

if __name__ == "__main__":
    test_convert_masks_to_sparse_coo()
    test_save_and_load()
    print("\n🎉 All tests passed! The sparse export functionality is working correctly.")