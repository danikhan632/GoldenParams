#!/usr/bin/env python3
"""
Analyze mask mismatches between a mask file and a model.
"""

import torch
import sys
from transformers import AutoModel, AutoModelForCausalLM
from golden_params.utils import convert_masks_to_torch_sparse_coo


def analyze_mismatches(model_name, mask_path):
    """Analyze mask shape mismatches."""
    print(f"Analyzing mask mismatches...")
    print(f"  Model: {model_name}")
    print(f"  Masks: {mask_path}")
    print()

    # Load model
    print("Loading model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load masks
    print("Loading masks...")
    try:
        mask_data = torch.load(mask_path, map_location='cpu')
        masks = mask_data.get("masks", mask_data.get("intersection_masks", {}))

        # Convert to sparse if needed
        if masks and not all(isinstance(m, torch.Tensor) and m.is_sparse for m in masks.values()):
            masks = convert_masks_to_torch_sparse_coo(masks)

        print(f"Loaded {len(masks)} masks")
    except Exception as e:
        print(f"Error loading masks: {e}")
        return

    # Get model parameters
    model_params = {name: param.shape for name, param in model.named_parameters()}
    print(f"Model has {len(model_params)} parameters")
    print()

    # Find matches and mismatches
    matches = []
    mismatches = []
    mask_only = []
    param_only = []

    for mask_name, mask in masks.items():
        mask_shape = mask.shape if hasattr(mask, 'shape') else mask.size()

        if mask_name in model_params:
            param_shape = model_params[mask_name]
            if mask_shape == param_shape:
                matches.append((mask_name, mask_shape))
            else:
                mismatches.append((mask_name, mask_shape, param_shape))
        else:
            mask_only.append((mask_name, mask_shape))

    for param_name, param_shape in model_params.items():
        if param_name not in masks:
            param_only.append((param_name, param_shape))

    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✓ Matching masks:        {len(matches):4d} ({len(matches)/len(masks)*100:.1f}%)")
    print(f"✗ Mismatched shapes:     {len(mismatches):4d} ({len(mismatches)/len(masks)*100:.1f}%)")
    print(f"  Masks without params:  {len(mask_only):4d}")
    print(f"  Params without masks:  {len(param_only):4d}")
    print("=" * 80)
    print()

    # Print mismatches
    if mismatches:
        print("SHAPE MISMATCHES (these will be skipped):")
        print("=" * 80)
        print(f"{'Parameter Name':<50} {'Mask Shape':<20} {'Param Shape':<20}")
        print("-" * 80)
        for param_name, mask_shape, param_shape in sorted(mismatches):
            print(f"{param_name:<50} {str(mask_shape):<20} {str(param_shape):<20}")
        print("=" * 80)
        print()

    # Print masks without params
    if mask_only:
        print(f"MASKS WITHOUT CORRESPONDING PARAMETERS ({len(mask_only)}):")
        print("=" * 80)
        for mask_name, mask_shape in sorted(mask_only)[:10]:  # Show first 10
            print(f"  {mask_name:<50} {str(mask_shape)}")
        if len(mask_only) > 10:
            print(f"  ... and {len(mask_only) - 10} more")
        print("=" * 80)
        print()

    # Print params without masks
    if param_only:
        print(f"PARAMETERS WITHOUT MASKS ({len(param_only)}):")
        print("=" * 80)
        for param_name, param_shape in sorted(param_only)[:10]:  # Show first 10
            print(f"  {param_name:<50} {str(param_shape)}")
        if len(param_only) > 10:
            print(f"  ... and {len(param_only) - 10} more")
        print("=" * 80)
        print()

    # Recommendations
    print("RECOMMENDATIONS:")
    print("=" * 80)
    if len(mismatches) == 0 and len(mask_only) == 0:
        print("✓ Perfect match! All masks align with model parameters.")
    elif len(mismatches) < len(matches) * 0.1:  # Less than 10% mismatch
        print(f"⚠ Minor mismatches detected ({len(mismatches)} parameters).")
        print(f"  These parameters will use unmasked_lr during training.")
        print(f"  This is OK if you're fine with standard learning rates for these params.")
    else:
        print(f"✗ Significant mismatches detected ({len(mismatches)} parameters).")
        print(f"  Consider regenerating masks for this model architecture.")
        print(f"  Current mask file appears to be from a different model.")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python analyze_mask_mismatch.py <model_name> <mask_file.pt>")
        print()
        print("Example:")
        print("  python analyze_mask_mismatch.py Shekswess/trlm-135m examples/golden_masks.pt")
        sys.exit(1)

    model_name = sys.argv[1]
    mask_path = sys.argv[2]

    analyze_mismatches(model_name, mask_path)
