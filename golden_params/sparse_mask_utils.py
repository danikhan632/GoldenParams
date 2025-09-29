"""
Sparse mask utilities for memory-efficient mask operations using COO tensors.
"""
import torch
from typing import Dict, Optional, Tuple, Union, List
import warnings


class SparseMaskManager:
    """
    Manages sparse COO tensor masks for memory-efficient masking operations.
    
    This class provides utilities to convert dense boolean masks to sparse COO format,
    perform boolean operations on sparse masks, and apply masks efficiently.
    """
    
    @staticmethod
    def dense_to_sparse_coo(dense_mask: torch.Tensor) -> torch.Tensor:
        """
        Convert dense boolean mask to sparse COO tensor.
        
        Args:
            dense_mask: Dense boolean tensor
            
        Returns:
            Sparse COO tensor containing only True indices
        """
        if not isinstance(dense_mask, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
            
        if dense_mask.dtype != torch.bool:
            dense_mask = dense_mask.bool()
        
        # Get indices where mask is True
        indices = dense_mask.nonzero(as_tuple=False).t()
        
        if indices.numel() == 0:
            # Empty mask - create sparse tensor with no indices
            return torch.sparse_coo_tensor(
                indices=torch.empty((dense_mask.dim(), 0), dtype=torch.long, device=dense_mask.device),
                values=torch.empty(0, dtype=torch.bool, device=dense_mask.device),
                size=dense_mask.shape,
                device=dense_mask.device
            ).coalesce()
        
        # Create sparse COO tensor
        values = torch.ones(indices.shape[1], dtype=torch.bool, device=dense_mask.device)
        sparse_mask = torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=dense_mask.shape,
            device=dense_mask.device
        ).coalesce()
        
        return sparse_mask
    
    @staticmethod
    def sparse_coo_to_dense(sparse_mask: torch.Tensor) -> torch.Tensor:
        """
        Convert sparse COO tensor back to dense boolean tensor.
        
        Args:
            sparse_mask: Sparse COO tensor
            
        Returns:
            Dense boolean tensor
        """
        return sparse_mask.to_dense().bool()
    
    @staticmethod
    def sparse_logical_or(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
        """
        Perform logical OR on two sparse COO masks.
        
        Args:
            mask1: First sparse COO mask
            mask2: Second sparse COO mask
            
        Returns:
            Sparse COO tensor representing mask1 | mask2
        """
        if not (mask1.is_sparse and mask2.is_sparse):
            raise ValueError("Both inputs must be sparse COO tensors")
        
        if mask1.shape != mask2.shape:
            raise ValueError("Masks must have the same shape")
        
        # Convert to dense, perform OR, convert back to sparse
        # This is actually more memory efficient for sparse OR than manual index manipulation
        # when the masks are not extremely sparse
        dense1 = mask1.to_dense().bool()
        dense2 = mask2.to_dense().bool()
        result_dense = dense1 | dense2
        
        return SparseMaskManager.dense_to_sparse_coo(result_dense)
    
    @staticmethod
    def sparse_logical_and(mask1: torch.Tensor, mask2: torch.Tensor) -> torch.Tensor:
        """
        Perform logical AND on two sparse COO masks.
        
        Args:
            mask1: First sparse COO mask  
            mask2: Second sparse COO mask
            
        Returns:
            Sparse COO tensor representing mask1 & mask2
        """
        if not (mask1.is_sparse and mask2.is_sparse):
            raise ValueError("Both inputs must be sparse COO tensors")
        
        if mask1.shape != mask2.shape:
            raise ValueError("Masks must have the same shape")
        
        # For AND operation on sparse tensors, we need to find intersection of indices
        # Convert to dense for now - could be optimized further with index set operations
        dense1 = mask1.to_dense().bool()
        dense2 = mask2.to_dense().bool()
        result_dense = dense1 & dense2
        
        return SparseMaskManager.dense_to_sparse_coo(result_dense)
    
    @staticmethod  
    def sparse_logical_not(mask: torch.Tensor) -> torch.Tensor:
        """
        Perform logical NOT on sparse COO mask.
        
        Note: This operation results in a mostly dense tensor and should be used sparingly.
        Consider redesigning logic to avoid NOT operations on sparse masks when possible.
        
        Args:
            mask: Sparse COO mask
            
        Returns:
            Sparse COO tensor representing ~mask
        """
        warnings.warn(
            "sparse_logical_not creates mostly dense tensors and may not save memory. "
            "Consider redesigning logic to avoid NOT operations on sparse masks.",
            UserWarning
        )
        
        if not mask.is_sparse:
            raise ValueError("Input must be a sparse COO tensor")
        
        dense_mask = mask.to_dense().bool()
        result_dense = ~dense_mask
        return SparseMaskManager.dense_to_sparse_coo(result_dense)
    
    @staticmethod
    def apply_sparse_mask(tensor: torch.Tensor, sparse_mask: torch.Tensor, fill_value: float = 0.0) -> torch.Tensor:
        """
        Apply sparse mask to tensor by setting non-masked values to fill_value.
        
        Args:
            tensor: Tensor to mask
            sparse_mask: Sparse COO mask (True values are kept, False values become fill_value)
            fill_value: Value to set for masked-out elements
            
        Returns:
            Masked tensor
        """
        if not sparse_mask.is_sparse:
            raise ValueError("Mask must be a sparse COO tensor")
        
        if tensor.shape != sparse_mask.shape:
            raise ValueError("Tensor and mask must have the same shape")
        
        # Create result tensor filled with fill_value
        result = torch.full_like(tensor, fill_value)
        
        # Get the indices and copy values from original tensor
        mask_indices = sparse_mask.indices()
        if mask_indices.shape[1] > 0:  # If mask has any True values
            # Use advanced indexing to set values
            if tensor.dim() == 1:
                result[mask_indices[0]] = tensor[mask_indices[0]]
            elif tensor.dim() == 2:
                result[mask_indices[0], mask_indices[1]] = tensor[mask_indices[0], mask_indices[1]]
            else:
                # For higher dimensions, use tuple indexing
                idx_tuple = tuple(mask_indices[i] for i in range(mask_indices.shape[0]))
                result[idx_tuple] = tensor[idx_tuple]
        
        return result
    
    @staticmethod
    def sparse_mask_density(sparse_mask: torch.Tensor) -> float:
        """
        Calculate the density (fraction of True values) of a sparse mask.
        
        Args:
            sparse_mask: Sparse COO mask
            
        Returns:
            Density as a float between 0 and 1
        """
        if not sparse_mask.is_sparse:
            raise ValueError("Input must be a sparse COO tensor")
        
        total_elements = sparse_mask.numel()
        if total_elements == 0:
            return 0.0
        
        true_elements = sparse_mask.coalesce().values().sum().item()
        return float(true_elements) / float(total_elements)
    
    @staticmethod
    def get_sparse_memory_usage(sparse_mask: torch.Tensor) -> Dict[str, int]:
        """
        Get memory usage statistics for sparse mask.
        
        Args:
            sparse_mask: Sparse COO mask
            
        Returns:
            Dictionary with memory usage information in bytes
        """
        if not sparse_mask.is_sparse:
            raise ValueError("Input must be a sparse COO tensor")
        
        sparse_mask = sparse_mask.coalesce()
        
        # Calculate memory usage
        indices_memory = sparse_mask.indices().element_size() * sparse_mask.indices().numel()
        values_memory = sparse_mask.values().element_size() * sparse_mask.values().numel()
        sparse_memory = indices_memory + values_memory
        
        # Compare with dense equivalent
        dense_memory = sparse_mask.numel() * torch.tensor([], dtype=torch.bool).element_size()
        
        return {
            'sparse_memory_bytes': sparse_memory,
            'dense_memory_bytes': dense_memory,
            'memory_savings_bytes': dense_memory - sparse_memory,
            'memory_savings_ratio': (dense_memory - sparse_memory) / dense_memory if dense_memory > 0 else 0.0
        }


class SparseGradientAccumulator:
    """
    Memory-efficient gradient accumulator using sparse masks.
    
    Instead of storing full gradient tensors, this accumulator stores only
    the sparse coordinates and their values, significantly reducing memory usage
    for sparse updates.
    """
    
    def __init__(self):
        self._accumulated_coords: Dict[str, torch.Tensor] = {}  # Sparse indices
        self._accumulated_values: Dict[str, torch.Tensor] = {}  # Corresponding values
        self._param_shapes: Dict[str, torch.Size] = {}
        self._devices: Dict[str, torch.device] = {}
    
    def add_sparse_gradient(self, name: str, param_shape: torch.Size, 
                          sparse_mask: torch.Tensor, gradient: torch.Tensor):
        """
        Add a sparse gradient update.
        
        Args:
            name: Parameter name
            param_shape: Shape of the parameter
            sparse_mask: Sparse COO mask indicating which coordinates to update
            gradient: Full gradient tensor (will be masked)
        """
        if not sparse_mask.is_sparse:
            raise ValueError("sparse_mask must be a sparse COO tensor")
        
        self._param_shapes[name] = param_shape
        self._devices[name] = gradient.device
        
        # Get masked gradient values
        masked_gradient = SparseMaskManager.apply_sparse_mask(gradient, sparse_mask)
        
        # Get sparse indices and values
        sparse_mask = sparse_mask.coalesce()
        indices = sparse_mask.indices()
        
        if indices.shape[1] > 0:  # If there are any True values
            # Extract values using advanced indexing
            if gradient.dim() == 1:
                values = masked_gradient[indices[0]]
            elif gradient.dim() == 2:
                values = masked_gradient[indices[0], indices[1]]
            else:
                idx_tuple = tuple(indices[i] for i in range(indices.shape[0]))
                values = masked_gradient[idx_tuple]
            
            if name not in self._accumulated_coords:
                self._accumulated_coords[name] = indices.clone()
                self._accumulated_values[name] = values.clone()
            else:
                # Combine with existing accumulation
                # For simplicity, convert to dense, add, convert back to sparse
                # Could be optimized with sparse arithmetic
                existing_sparse = torch.sparse_coo_tensor(
                    self._accumulated_coords[name],
                    self._accumulated_values[name],
                    param_shape,
                    device=gradient.device
                ).coalesce()
                
                new_sparse = torch.sparse_coo_tensor(
                    indices,
                    values,
                    param_shape,
                    device=gradient.device
                ).coalesce()
                
                # Add the sparse tensors
                combined_dense = existing_sparse.to_dense() + new_sparse.to_dense()
                combined_sparse = SparseMaskManager.dense_to_sparse_coo(combined_dense.bool()) 
                
                # Store the result
                combined_sparse = combined_sparse.coalesce()
                self._accumulated_coords[name] = combined_sparse.indices()
                
                # Get the actual summed values from dense addition
                if combined_sparse.indices().shape[1] > 0:
                    if combined_dense.dim() == 1:
                        self._accumulated_values[name] = combined_dense[combined_sparse.indices()[0]]
                    elif combined_dense.dim() == 2:
                        self._accumulated_values[name] = combined_dense[
                            combined_sparse.indices()[0], combined_sparse.indices()[1]
                        ]
                    else:
                        idx_tuple = tuple(combined_sparse.indices()[i] for i in range(combined_sparse.indices().shape[0]))
                        self._accumulated_values[name] = combined_dense[idx_tuple]
                else:
                    self._accumulated_values[name] = torch.empty(0, dtype=gradient.dtype, device=gradient.device)
    
    def get_accumulated_gradient(self, name: str) -> Optional[torch.Tensor]:
        """
        Get the accumulated gradient as a dense tensor.
        
        Args:
            name: Parameter name
            
        Returns:
            Accumulated gradient tensor, or None if no accumulation exists
        """
        if name not in self._accumulated_coords:
            return None
        
        # Reconstruct sparse tensor and convert to dense
        sparse_grad = torch.sparse_coo_tensor(
            self._accumulated_coords[name],
            self._accumulated_values[name],
            self._param_shapes[name],
            device=self._devices[name]
        ).coalesce()
        
        return sparse_grad.to_dense()
    
    def clear(self):
        """Clear all accumulated gradients."""
        self._accumulated_coords.clear()
        self._accumulated_values.clear()
        self._param_shapes.clear()
        self._devices.clear()
    
    def get_memory_usage(self) -> Dict[str, int]:
        """
        Get memory usage statistics for the accumulator.
        
        Returns:
            Dictionary with memory usage information
        """
        total_sparse_memory = 0
        total_dense_memory = 0
        
        for name in self._accumulated_coords:
            coords_memory = (self._accumulated_coords[name].element_size() * 
                           self._accumulated_coords[name].numel())
            values_memory = (self._accumulated_values[name].element_size() * 
                           self._accumulated_values[name].numel())
            sparse_memory = coords_memory + values_memory
            
            dense_memory = torch.tensor(0.0).new_zeros(self._param_shapes[name]).element_size() * torch.tensor(self._param_shapes[name]).prod().item()
            
            total_sparse_memory += sparse_memory
            total_dense_memory += dense_memory
        
        return {
            'total_sparse_memory_bytes': total_sparse_memory,
            'total_dense_memory_bytes': total_dense_memory,
            'memory_savings_bytes': total_dense_memory - total_sparse_memory,
            'memory_savings_ratio': (total_dense_memory - total_sparse_memory) / total_dense_memory if total_dense_memory > 0 else 0.0
        }