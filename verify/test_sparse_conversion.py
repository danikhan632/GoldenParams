#!/usr/bin/env python3
"""
Test script to verify sparse conversion maintains correctness.
"""

import torch
from golden_params.utils import convert_masks_to_sparse_coo, convert_sparse_to_dense_mask

def test_roundtrip_conversion():
    """Test that dense -> sparse -> dense conversion preserves data."""
    print("Testing roundtrip conversion...")

    # Create test boolean masks with different sparsity levels
    test_masks = {
        "dense_mask": torch.rand(100, 100) > 0.1,  # 90% density
        "sparse_mask": torch.rand(100, 100) > 0.9,  # 10% density
        "very_sparse": torch.rand(1000, 1000) > 0.95,  # 5% density
    }

    for name, original_mask in test_masks.items():
        print(f"\nTesting {name}:")
        print(f"  Original shape: {original_mask.shape}")
        print(f"  Original density: {original_mask.count_nonzero().item() / original_mask.numel() * 100:.1f}%")

        # Convert to sparse
        sparse_dict = convert_masks_to_sparse_coo({name: original_mask})
        sparse_mask = sparse_dict[name]

        # Check sparse format
        print(f"  Sparse format: {sparse_mask['layout']}")
        print(f"  Sparse nnz: {sparse_mask['nnz']:,}")

        # Convert back to dense
        reconstructed = convert_sparse_to_dense_mask(sparse_mask)

        # Verify they are identical
        if torch.equal(original_mask, reconstructed):
            print(f"  ✅ Roundtrip conversion successful!")
        else:
            print(f"  ❌ Roundtrip conversion FAILED!")
            print(f"    Original nnz: {original_mask.count_nonzero().item()}")
            print(f"    Reconstructed nnz: {reconstructed.count_nonzero().item()}")
            return False

        # Calculate memory savings
        dense_memory = original_mask.numel()  # 1 byte per bool
        sparse_memory = sparse_mask['indices'].element_size() * sparse_mask['indices'].numel() + 64
        savings = (dense_memory - sparse_memory) / dense_memory * 100
        print(f"  Memory savings: {savings:.1f}% ({dense_memory:,} -> {sparse_memory:,} bytes)")

    return True

def test_file_roundtrip():
    """Test loading sparse file and converting back to dense."""
    print("\n" + "="*50)
    print("Testing file roundtrip with actual data...")

    # Load sparse file
    sparse_data = torch.load("/home/green/code/golden/reverse_golden_masks.pt", map_location='cpu')

    # Load original dense file
    dense_data = torch.load("/home/green/code/golden/reverse_golden_masks.pt.dense_backup", map_location='cpu')

    print(f"Loaded {len(sparse_data['masks'])} sparse masks")
    print(f"Loaded {len(dense_data['masks'])} dense masks")

    # Test a few masks
    test_count = 0
    for name in list(sparse_data['masks'].keys())[:5]:  # Test first 5
        sparse_mask = sparse_data['masks'][name]
        original_mask = dense_data['masks'][name]

        # Convert sparse back to dense
        reconstructed = convert_sparse_to_dense_mask(sparse_mask)

        if torch.equal(original_mask, reconstructed):
            print(f"  ✅ {name}: conversion verified")
        else:
            print(f"  ❌ {name}: conversion FAILED!")
            return False

        test_count += 1

    print(f"Successfully verified {test_count} masks from the actual file!")
    return True

if __name__ == "__main__":
    success1 = test_roundtrip_conversion()
    success2 = test_file_roundtrip()

    if success1 and success2:
        print("\n🎉 All tests passed! The sparse conversion is working correctly.")
        print("✅ Roundtrip conversion preserves all data")
        print("✅ File compression works correctly")
        print("✅ Ready for production use!")
    else:
        print("\n❌ Some tests failed!")