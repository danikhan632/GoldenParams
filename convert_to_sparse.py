#!/usr/bin/env python3
"""
Utility to convert existing dense mask files to sparse COO format.
"""

import torch
import os
import shutil
from golden_params.utils import convert_masks_to_sparse_coo

def convert_dense_to_sparse(input_path, output_path=None):
    """Convert a dense mask file to sparse COO format."""

    if output_path is None:
        # Create backup and use original name
        backup_path = input_path + ".dense_backup"
        shutil.copy2(input_path, backup_path)
        output_path = input_path
        print(f"Created backup: {backup_path}")

    print(f"Loading dense masks from: {input_path}")

    # Load the original file
    data = torch.load(input_path, map_location='cpu')

    # Analyze original file
    original_size = os.path.getsize(input_path)
    print(f"Original file size: {original_size / (1024**2):.1f} MB")

    if 'masks' not in data:
        print("No 'masks' key found in file!")
        return

    masks = data['masks']

    # Check if already sparse
    first_mask = next(iter(masks.values()))
    if hasattr(first_mask, 'is_sparse') and first_mask.is_sparse:
        print("Masks are already sparse!")
        return

    # Count parameters and sparsity
    total_params = 0
    total_nonzero = 0

    print("\nAnalyzing masks...")
    for name, mask in masks.items():
        total = mask.numel()
        nonzero = mask.count_nonzero().item()
        total_params += total
        total_nonzero += nonzero

    sparsity = (1 - total_nonzero / total_params) * 100
    print(f"Total parameters: {total_params:,}")
    print(f"Non-zero parameters: {total_nonzero:,}")
    print(f"Sparsity: {sparsity:.2f}%")

    # Convert to sparse
    print("\nConverting to sparse COO tensors...")
    sparse_masks = convert_masks_to_sparse_coo(masks)

    # Update data dictionary
    data['masks'] = sparse_masks

    # Save sparse version
    print(f"Saving sparse masks to: {output_path}")
    torch.save(data, output_path)

    # Analyze new file
    new_size = os.path.getsize(output_path)
    reduction = (original_size - new_size) / original_size * 100

    print(f"\nResults:")
    print(f"Original size: {original_size / (1024**2):.1f} MB")
    print(f"New size: {new_size / (1024**2):.1f} MB")
    print(f"Size reduction: {reduction:.1f}%")
    print(f"Compression ratio: {original_size / new_size:.1f}x")

def verify_sparse_file(file_path):
    """Verify that a file contains sparse tensors and analyze memory usage."""
    print(f"\nVerifying sparse file: {file_path}")

    data = torch.load(file_path, map_location='cpu')
    masks = data['masks']

    total_sparse_memory = 0
    total_dense_equivalent = 0

    sample_count = 0
    for name, mask in masks.items():
        # Check if it's our compact sparse format
        if isinstance(mask, dict) and mask.get('layout') == 'compact_sparse':
            # Calculate memory usage for compact format
            indices_memory = mask['indices'].element_size() * mask['indices'].numel()
            metadata_memory = 64  # Rough estimate for metadata
            sparse_memory = indices_memory + metadata_memory

            total_elements = torch.prod(torch.tensor(mask['shape'])).item()
            dense_equivalent = total_elements * 1  # 1 byte per boolean

            if sample_count < 3:
                print(f"  {name}: {sparse_memory:,} bytes (vs {dense_equivalent:,} dense), nnz={mask['nnz']:,}")
                print(f"    sparsity: {(1 - mask['nnz']/total_elements)*100:.1f}%")
                sample_count += 1

        elif hasattr(mask, 'is_sparse') and mask.is_sparse:
            # Standard sparse tensor
            indices_memory = mask.indices().element_size() * mask.indices().numel()
            values_memory = mask.values().element_size() * mask.values().numel()
            sparse_memory = indices_memory + values_memory
            dense_equivalent = mask.numel() * 1  # 1 byte per boolean

            if sample_count < 3:
                print(f"  {name}: {sparse_memory:,} bytes (vs {dense_equivalent:,} dense)")
                sample_count += 1
        else:
            print(f"ERROR: {name} is not in sparse format!")
            return False

        total_sparse_memory += sparse_memory
        total_dense_equivalent += dense_equivalent

    print(f"\nTotal sparse memory: {total_sparse_memory / (1024**2):.1f} MB")
    print(f"Equivalent dense memory: {total_dense_equivalent / (1024**2):.1f} MB")
    print(f"Memory efficiency: {total_sparse_memory / total_dense_equivalent * 100:.1f}% of dense")
    print(f"Compression ratio: {total_dense_equivalent / total_sparse_memory:.1f}x")

    return True

if __name__ == "__main__":
    input_file = "/home/green/code/golden/reverse_golden_masks.pt"

    if os.path.exists(input_file):
        convert_dense_to_sparse(input_file)
        verify_sparse_file(input_file)
    else:
        print(f"File not found: {input_file}")