#!/usr/bin/env python3
"""
Diagnostic script to check the structure of a golden mask file.
"""

import torch
import sys

def check_mask_file(mask_path):
    """Check the structure of a mask file and report details."""
    print(f"Loading mask file: {mask_path}")
    print("=" * 60)

    try:
        mask_data = torch.load(mask_path, map_location='cpu')
        print(f"✓ Successfully loaded mask file")
        print(f"  Type: {type(mask_data)}")

        if isinstance(mask_data, dict):
            print(f"  Keys: {list(mask_data.keys())}")

            # Check for common mask keys
            masks = mask_data.get("masks", mask_data.get("intersection_masks", None))

            if masks is None:
                print("\n⚠ Warning: No 'masks' or 'intersection_masks' key found!")
                print("  Available keys:", list(mask_data.keys()))
            else:
                print(f"\n✓ Found masks under key: {'masks' if 'masks' in mask_data else 'intersection_masks'}")
                print(f"  Number of masked parameters: {len(masks)}")

                # Sample a few masks to check their structure
                print("\nSample mask inspection (first 3):")
                for i, (name, mask) in enumerate(list(masks.items())[:3]):
                    print(f"\n  {i+1}. {name}")
                    print(f"     Type: {type(mask)}")

                    if isinstance(mask, torch.Tensor):
                        print(f"     Shape: {mask.shape}")
                        print(f"     Dtype: {mask.dtype}")
                        print(f"     Is sparse: {mask.is_sparse}")
                        print(f"     Layout: {mask.layout}")

                        if mask.is_sparse:
                            print(f"     Sparse nnz: {mask._nnz()}")
                            dense = mask.to_dense()
                            if dense.dtype == torch.bool:
                                density = dense.sum().item() / dense.numel()
                            else:
                                density = (dense != 0).sum().item() / dense.numel()
                            print(f"     Density: {density:.2%}")
                        else:
                            if mask.dtype == torch.bool:
                                density = mask.sum().item() / mask.numel()
                            else:
                                density = (mask != 0).sum().item() / mask.numel()
                            print(f"     Density: {density:.2%}")
                    else:
                        print(f"     ⚠ Not a tensor!")

                # Check if all masks are tensors
                all_tensors = all(isinstance(m, torch.Tensor) for m in masks.values())
                print(f"\n{'✓' if all_tensors else '✗'} All masks are tensors: {all_tensors}")

                if all_tensors:
                    # Check if all are sparse
                    all_sparse = all(m.is_sparse for m in masks.values())
                    print(f"{'✓' if all_sparse else '✗'} All masks are sparse: {all_sparse}")

                    # Check dtypes
                    dtypes = set(m.dtype for m in masks.values())
                    print(f"  Mask dtypes found: {dtypes}")

            # Check for summary info
            if "summary" in mask_data:
                print(f"\n✓ Found summary information")

            if "kl_divergence" in mask_data:
                print(f"✓ Found KL divergence: {mask_data['kl_divergence']}")

        else:
            print(f"⚠ Warning: Expected dict, got {type(mask_data)}")

    except Exception as e:
        print(f"✗ Error loading mask file: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_mask_file.py <path_to_mask_file.pt>")
        sys.exit(1)

    mask_path = sys.argv[1]
    check_mask_file(mask_path)
