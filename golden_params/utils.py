import torch
from contextlib import contextmanager, nullcontext

def printc(obj, color="cyan"):


    color_code = {
        "black": "30", "red": "31", "green": "32", "yellow": "33",
        "blue": "34", "magenta": "35", "cyan": "36", "white": "37"
    }

    # Determine ANSI color code
    if isinstance(color, int) and 30 <= color <= 37:
        code = str(color)
    elif isinstance(color, str) and color in color_code:
        code = color_code[color]
    else:
        code = None

    colored_text = f"\033[{code}m{obj}\033[0m" if code else str(obj)
    print(colored_text)

@contextmanager
def eval_mode(model: torch.nn.Module):
    """Temporarily sets ``model`` to evaluation mode to disable dropout."""
    was_training = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(was_training)

def forward_pass_logprobs_for_fixed_ids(
    model_obj: torch.nn.Module,
    query_responses: torch.LongTensor,
    query_only: torch.LongTensor,
    pad_token_id: int,
    enable_grads: bool = False,
    target_device: torch.device = None,
) -> torch.FloatTensor:

    if query_responses.dim() == 1:
        query_responses = query_responses.unsqueeze(0)
    if query_only.dim() == 1:
        query_only = query_only.unsqueeze(0)

    unwrapped = model_obj
    if hasattr(model_obj, "module"):
        unwrapped = model_obj.module

    # Determine the device to run on. In DDP/sequence-parallel runs each rank
    # owns a single device; ensure all inputs are moved to that device.
    if target_device is None:
        try:
            model_device = next(unwrapped.parameters()).device
        except StopIteration:
            # Fallback if model has no parameters (unlikely)
            model_device = torch.device("cpu")
    else:
        model_device = target_device

    # Ensure incoming tensors are on the same device as the model for this rank.
    query_responses = query_responses.to(model_device, non_blocking=True)
    query_only = query_only.to(model_device, non_blocking=True)

    # Use context manager to control gradient calculation
    grad_context = nullcontext() if enable_grads else torch.no_grad()

    with eval_mode(unwrapped), grad_context:
        attention_mask = query_responses != pad_token_id
        prompt_mask = torch.zeros_like(query_responses, dtype=torch.bool)
        prompt_mask[:, : query_only.shape[1]] = query_only != pad_token_id

        position_ids = attention_mask.cumsum(dim=1) - attention_mask.long()
        input_ids = torch.where(
            attention_mask,
            query_responses,
            torch.full_like(query_responses, pad_token_id)
        )

        # Make sure derived tensors are also on the correct device
        attention_mask = attention_mask.to(model_device, non_blocking=True)
        position_ids = position_ids.to(model_device, non_blocking=True)
        input_ids = input_ids.to(model_device, non_blocking=True)

    outputs = unwrapped(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        )

    # Align logits with completion tokens only: for token t, logits predict token t+1.
    # Keep positions from prompt_len-1 (predicting first completion token) to last-1.
    total_len = input_ids.shape[1]
    prompt_len = query_only.shape[1]
    # Guard against degenerate cases
    start = max(0, prompt_len - 1)
    end = total_len - 1
    next_logits = outputs.logits[:, start:end, :]

    logprobs = torch.log_softmax(next_logits, dim=-1)


    # clamp any NaNs or Infs to a large negative value
    logprobs = torch.nan_to_num(
        logprobs,
        nan=-1e9,
        posinf=-1e9,
        neginf=-1e9
    )

    return logprobs


def get_kl_divergence(ref_log_probs: torch.Tensor, z_log_probs: torch.Tensor) -> torch.Tensor:
    """Computes the KL divergence ``KL(ref || z)`` between two distributions
    given as log-probabilities.

    Args:
        ref_log_probs: Tensor of shape ``[B, T, V]`` – log-probs from the
            reference model.
        z_log_probs:   Tensor of shape ``[B, T, V]`` – log-probs from the target
            model.

    Returns:
        Tensor of shape ``[B, T]`` containing the per-token KL divergence.
    """
    # Ensure both tensors are on the same device for distributed/multi-GPU runs
    # Prefer the device of ref_log_probs if it's CUDA; otherwise use z_log_probs' device
    common_device = (
        ref_log_probs.device if ref_log_probs.is_cuda else z_log_probs.device
    )
    if ref_log_probs.device != common_device:
        ref_log_probs = ref_log_probs.to(common_device, non_blocking=True)
    if z_log_probs.device != common_device:
        z_log_probs = z_log_probs.to(common_device, non_blocking=True)

    ref_probs = ref_log_probs.exp()
    kl = ref_probs * (ref_log_probs - z_log_probs)
    return kl.sum(dim=-1)


def convert_masks_to_sparse_coo(masks_dict):
    """
    Convert a dictionary of boolean masks to an efficient sparse format.

    Instead of using torch.sparse_coo_tensor (which has large overhead for boolean data),
    this stores only the indices of True values in a compact format.

    Args:
        masks_dict: Dictionary mapping parameter names to boolean mask tensors

    Returns:
        Dictionary mapping parameter names to compact sparse representation
    """
    sparse_masks = {}
    for name, mask in masks_dict.items():
        if not isinstance(mask, torch.Tensor):
            raise ValueError(f"Mask for {name} must be a torch.Tensor")

        if mask.dtype != torch.bool:
            raise ValueError(f"Mask for {name} must be boolean tensor")

        # Get linear indices where mask is True (much more compact)
        true_indices = torch.nonzero(mask.flatten(), as_tuple=True)[0]

        # Store in compact format with metadata
        sparse_mask = {
            'indices': true_indices.to(dtype=torch.int32),  # Use int32 instead of int64 to save space
            'shape': mask.shape,
            'nnz': true_indices.numel(),
            'dtype': 'bool',
            'layout': 'compact_sparse'
        }

        sparse_masks[name] = sparse_mask

    return sparse_masks


def convert_masks_to_torch_sparse_coo(masks_dict):
    """
    Convert a dictionary of boolean masks to PyTorch sparse COO tensors.

    This is specifically for use with the MaskedAdamW optimizer which requires
    actual torch.sparse_coo_tensor objects.

    Args:
        masks_dict: Dictionary mapping parameter names to boolean mask tensors

    Returns:
        Dictionary mapping parameter names to sparse COO tensors
    """
    sparse_masks = {}
    for name, mask in masks_dict.items():
        if not isinstance(mask, torch.Tensor):
            raise ValueError(f"Mask for {name} must be a torch.Tensor")

        if mask.dtype != torch.bool:
            raise ValueError(f"Mask for {name} must be boolean tensor")

        # Get indices where mask is True
        indices = torch.nonzero(mask, as_tuple=False).t()

        # Create values tensor (all ones for boolean masks)
        values = torch.ones(indices.shape[1], dtype=torch.float32, device=mask.device)

        # Create sparse COO tensor
        sparse_mask = torch.sparse_coo_tensor(
            indices, values, mask.shape, device=mask.device, dtype=torch.float32
        ).coalesce()

        sparse_masks[name] = sparse_mask

    return sparse_masks


def convert_sparse_to_dense_mask(sparse_mask):
    """
    Convert compact sparse representation back to dense boolean mask.

    Args:
        sparse_mask: Compact sparse representation from convert_masks_to_sparse_coo

    Returns:
        Dense boolean tensor
    """
    if isinstance(sparse_mask, dict) and sparse_mask.get('layout') == 'compact_sparse':
        # Handle our compact format
        indices = sparse_mask['indices']
        shape = sparse_mask['shape']

        # Create dense boolean tensor
        mask = torch.zeros(torch.prod(torch.tensor(shape)), dtype=torch.bool)
        mask[indices.long()] = True  # Convert back to long for indexing
        return mask.view(shape)

    elif hasattr(sparse_mask, 'is_sparse') and sparse_mask.is_sparse:
        # Handle standard sparse COO tensors
        return sparse_mask.to_dense().bool()

    else:
        # Already dense
        return sparse_mask
