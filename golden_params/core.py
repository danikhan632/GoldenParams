"""
Core functionality for golden parameters computation.

This module contains the standalone implementation of reverse golden parameters
from the original example_goldy.py file.
"""

import os
import json
import math
import random
from itertools import islice
from typing import Any, Callable, Dict, List, Optional, Union
from contextlib import contextmanager, nullcontext
import copy

import torch
from transformers import (
    PreTrainedTokenizerBase,
)

from .utils import (
    printc,
    forward_pass_logprobs_for_fixed_ids,
    get_kl_divergence,
    convert_masks_to_sparse_coo,
)


@contextmanager
def eval_mode(model):
    """Context manager to temporarily set model to eval mode."""
    prev = model.training
    model.eval()
    try:
        yield
    finally:
        model.train(prev)


def get_reverse_golden_params(
    model,
    eval_dataset,
    prompt_func: Callable[[List[Any]], tuple[Dict[str, torch.Tensor], Any]],
    processing_class,
    top_k_percent: float = 30.0,
    chunk_size: int = 4,
    inflation_step: float = 1.2,
    max_inflation_factor: float = 2.0,
    decay_factor: float = 0.9,
    min_delta: float = 0.01,
    save_path: Optional[str] = None,
    num_samples: int = 4,
    batch_size: int = 1,
    eval_sample_size: int = 8,
    max_stagnant_iterations: int = 5,
    fluctuation_strategy: str = "adaptive_sinusoidal",
    fluctuation_amplitude: float = 0.3,
    fluctuation_frequency: float = 0.1,
    momentum_factor: float = 0.8,
    exploration_temperature: float = 1.0,
    pad_token_id: int = 4,
) -> Dict[str, Any]:
    """Adaptive reverse Golden Parameter search (standalone version)"""
    from math import ceil

    if eval_dataset is None or len(eval_dataset) == 0:
        return {"masks": {}, "summary": {}, "kl_divergence": 0.0}

    # Determine actual number of samples to use
    actual_num_samples = min(num_samples, len(eval_dataset))

    target_pct = float(max(0.0, min(100.0, top_k_percent)))
    if target_pct <= 0.0:
        return {"masks": {}, "summary": {}, "kl_divergence": 0.0}

    effective_pct = target_pct
    max_pct = target_pct * max_inflation_factor
    min_pct = target_pct * 0.5  # Lower bound for fluctuation
    device = next(model.parameters()).device

    model.train()

    flat_sums: Dict[str, torch.Tensor] = {}
    shapes: Dict[str, torch.Size] = {}
    num_processed = 0
    best_kl = -1.0
    stagnant_iterations = 0

    # Sophisticated fluctuation state variables
    momentum_velocity = 0.0
    fluctuation_phase = 0.0
    kl_history = []
    temperature = exploration_temperature
    oscillator_velocity = 0.0
    chaos_state = random.random()

    def calculate_sophisticated_fluctuation(iteration: int, current_pct: float, kl_val: float, improvement: float) -> float:
        """Calculate sophisticated fluctuation using various strategies"""
        nonlocal momentum_velocity, fluctuation_phase, temperature, oscillator_velocity, chaos_state

        if fluctuation_strategy == "adaptive_sinusoidal":
            # Sinusoidal fluctuation with adaptive amplitude based on performance
            amplitude_factor = fluctuation_amplitude * (1.0 + math.tanh(improvement * 10))
            sine_component = amplitude_factor * math.sin(fluctuation_phase + iteration * fluctuation_frequency * 2 * math.pi)
            fluctuation_phase += fluctuation_frequency * 2 * math.pi
            base_pct = target_pct + (max_pct - target_pct) * 0.5
            return max(min_pct, min(max_pct, base_pct + sine_component * (max_pct - min_pct)))

        elif fluctuation_strategy == "momentum_based":
            # Momentum-based fluctuation with inertia
            target_direction = 1.0 if improvement > min_delta else -1.0
            momentum_velocity = momentum_factor * momentum_velocity + (1 - momentum_factor) * target_direction
            delta = momentum_velocity * fluctuation_amplitude * (max_pct - min_pct) * 0.1
            return max(min_pct, min(max_pct, current_pct + delta))

        elif fluctuation_strategy == "exploration_decay":
            # Temperature-based exploration with decay
            temperature *= 0.995  # Gradual cooling
            exploration_factor = math.exp(-abs(improvement) / (temperature + 1e-8))
            noise = (random.random() - 0.5) * exploration_factor * fluctuation_amplitude
            return max(min_pct, min(max_pct, target_pct + noise * (max_pct - min_pct)))

        elif fluctuation_strategy == "harmonic_oscillator":
            # Damped harmonic oscillator with spring-like behavior
            spring_constant = 0.1
            damping = 0.05
            force = -spring_constant * (current_pct - target_pct) - damping * oscillator_velocity
            oscillator_velocity += force
            return max(min_pct, min(max_pct, current_pct + oscillator_velocity))

        elif fluctuation_strategy == "chaos_driven":
            # Chaotic fluctuation using logistic map
            chaos_param = 3.8  # Chaos parameter
            chaos_state = chaos_param * chaos_state * (1 - chaos_state)
            chaotic_offset = (chaos_state - 0.5) * fluctuation_amplitude * (max_pct - min_pct)
            return max(min_pct, min(max_pct, target_pct + chaotic_offset))

        else:
            # Fallback to original strategy
            if improvement > min_delta:
                return max(target_pct, current_pct * decay_factor)
            else:
                return min(max_pct, current_pct * inflation_step)

    # Precompute target counts for later pruning
    target_counts: Dict[str, int] = {}
    for name, p in model.named_parameters():
        shapes[name] = p.shape
        target_counts[name] = ceil((target_pct / 100.0) * p.numel())

    def build_masks(flat_scores: Dict[str, torch.Tensor], pct: float) -> Dict[str, torch.Tensor]:
        masks: Dict[str, torch.Tensor] = {}
        for name, scores in flat_scores.items():
            numel = scores.numel()
            if pct >= 100.0:
                mask = torch.ones(numel, dtype=torch.bool, device=device)
            else:
                k = max(1, int(ceil((pct / 100.0) * numel)))
                idx = torch.topk(scores, k, largest=True, sorted=False).indices
                mask = torch.zeros(numel, dtype=torch.bool, device=device)
                mask.scatter_(0, idx, True)
            masks[name] = mask.view(shapes[name])
        return masks

    def eval_kl(masks: Dict[str, torch.Tensor]) -> float:
        # Convert masks to sparse COO format and store original values sparsely
        sparse_masks = {}
        original_values = {}

        total_masked_params = 0
        with torch.no_grad():
            for name, mask in masks.items():
                if name in [n for n, p in model.named_parameters()]:
                    # Get indices where mask is True
                    indices = mask.nonzero(as_tuple=False).t()
                    if indices.numel() > 0:
                        total_masked_params += indices.shape[1]
                        # Create sparse COO tensor
                        sparse_masks[name] = torch.sparse_coo_tensor(
                            indices,
                            torch.ones(indices.shape[1], device=device, dtype=torch.bool),
                            mask.shape,
                            device=device
                        ).coalesce()

                        # Store only the values that will be modified (sparse storage)
                        param = dict(model.named_parameters())[name]
                        original_values[name] = param.data[mask].clone()

                        # Zero out the masked parameters
                        param.data[mask] = 0

        printc(f"    Masked {total_masked_params:,} parameters across {len(sparse_masks)} layers", "yellow")

        kl_total = 0.0
        kl_count = 0

        # Use specified sample size for KL evaluation
        printc(f"    Evaluating KL on {eval_sample_size} samples...", "yellow")

        # Process evaluation in batches - ensure we use different samples each time
        import random
        eval_indices = list(range(min(len(eval_dataset), eval_sample_size)))
        random.shuffle(eval_indices)  # Randomize evaluation samples

        for i in range(0, len(eval_indices), batch_size):
            if kl_count >= eval_sample_size:
                break

            batch_indices = eval_indices[i:i + batch_size]
            subset_slice = [eval_dataset[idx] for idx in batch_indices]

            # Convert to list
            try:
                subset = list(subset_slice)
            except Exception as e:
                subset = []
                for item in subset_slice:
                    subset.append(item)

            if len(subset) == 0:
                break

            query_only, query_responses = prompt_func(subset, processing_class)

            # Clear cache to free memory
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()

            with eval_mode(model):
                # Get perturbed logprobs with zeroed parameters
                perturbed = forward_pass_logprobs_for_fixed_ids(
                    model,
                    query_responses,
                    query_only,
                    pad_token_id=pad_token_id,
                    enable_grads=False,
                    target_device=device,
                )

            # Restore original values temporarily using sparse masks
            with torch.no_grad():
                for name, sparse_mask in sparse_masks.items():
                    if name in original_values:
                        param = dict(model.named_parameters())[name]
                        mask = sparse_mask.to_dense().bool()
                        param.data[mask] = original_values[name]

            with eval_mode(model):
                # Get original logprobs
                orig = forward_pass_logprobs_for_fixed_ids(
                    model,
                    query_responses,
                    query_only,
                    pad_token_id=pad_token_id,
                    enable_grads=False,
                    target_device=device,
                )

            # Zero out again for next iteration
            with torch.no_grad():
                for name, sparse_mask in sparse_masks.items():
                    param = dict(model.named_parameters())[name]
                    mask = sparse_mask.to_dense().bool()
                    param.data[mask] = 0

            # Calculate KL divergence with more debugging
            kl_tensor = get_kl_divergence(orig, perturbed)
            kl_val = kl_tensor.mean().item()

            # Debug: Print more detailed KL info
            if kl_count < 3:  # Only for first few batches to avoid spam
                printc(f"      Batch {kl_count}: KL = {kl_val:.6f}, orig_shape: {orig.shape}, perturbed_shape: {perturbed.shape}", "cyan")
                printc(f"      Orig mean: {orig.mean().item():.6f}, Perturbed mean: {perturbed.mean().item():.6f}", "cyan")
                printc(f"      Orig std: {orig.std().item():.6f}, Perturbed std: {perturbed.std().item():.6f}", "cyan")
                kl_std = kl_tensor.std().item() if kl_tensor.numel() > 1 else 0.0
                printc(f"      KL std: {kl_std:.6f}, KL min: {kl_tensor.min().item():.6f}, KL max: {kl_tensor.max().item():.6f}", "cyan")

            kl_total += kl_val
            kl_count += 1

        # Restore all original values using sparse storage
        with torch.no_grad():
            for name, sparse_mask in sparse_masks.items():
                if name in original_values:
                    param = dict(model.named_parameters())[name]
                    mask = sparse_mask.to_dense().bool()
                    param.data[mask] = original_values[name]

        return kl_total / kl_count if kl_count > 0 else 0.0

    iteration = 0
    printc(f"Starting reverse golden param search: target={target_pct:.1f}%, samples={actual_num_samples}", "blue")

    while num_processed < actual_num_samples:
        iteration += 1
        chunk = eval_dataset[num_processed:num_processed + chunk_size]
        if len(chunk) == 0:
            break

        printc(f"Iteration {iteration}: Processing chunk of {len(chunk)} samples (total processed: {num_processed}/{actual_num_samples})", "cyan")

        # Memory info before processing
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
            printc(f"  GPU Memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved", "yellow")

        # Process chunk in batches
        try:
            chunk_list = list(chunk)
        except Exception as e:
            printc(f"  Warning: Direct list conversion failed ({e}), trying manual iteration", "yellow")
            chunk_list = []
            for item in chunk:
                chunk_list.append(item)

        for batch_start in range(0, len(chunk_list), batch_size):
            batch_end = min(batch_start + batch_size, len(chunk_list))
            batch = chunk_list[batch_start:batch_end]

            model.zero_grad(set_to_none=True)
            query_only, query_responses = prompt_func(batch, processing_class)
            logprobs = forward_pass_logprobs_for_fixed_ids(
                model,
                query_responses,
                query_only,
                pad_token_id=pad_token_id,
                enable_grads=True,
                target_device=device,
            )
            fake_loss = logprobs.mean() * 1000.0
            fake_loss.backward()

            grad_params_updated = 0
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if (param.grad is None) or (not param.requires_grad):
                        continue
                    g = param.grad.detach()
                    if g.numel() == 0:
                        continue
                    flat_g = g.abs().reshape(-1)
                    if name not in flat_sums:
                        flat_sums[name] = flat_g
                    else:
                        flat_sums[name] += flat_g
                    grad_params_updated += 1

            if batch_start == 0:  # Log for first batch in chunk
                printc(f"  Batch size: {len(batch)}, Updated gradients for {grad_params_updated} parameters", "green")

        num_processed += len(chunk_list)

        printc(f"  Building candidate masks with effective_pct={effective_pct:.1f}%", "magenta")
        candidate_masks = build_masks(flat_sums, effective_pct)

        # Log mask statistics
        total_selected = sum(m.count_nonzero().item() for m in candidate_masks.values())
        total_params = sum(m.numel() for m in candidate_masks.values())
        actual_sparsity = (total_selected / total_params * 100) if total_params > 0 else 0
        printc(f"  Candidate masks: {total_selected:,}/{total_params:,} params ({actual_sparsity:.2f}%)", "green")

        printc(f"  Evaluating KL divergence...", "cyan")
        kl_val = eval_kl(candidate_masks)
        printc(f"  KL divergence: {kl_val:.6f} (best so far: {best_kl:.6f})", "blue")

        # Sophisticated adaptive percentage adjustment
        improvement = kl_val - best_kl
        kl_history.append(kl_val)

        # Update best KL if we found improvement
        if improvement > min_delta:
            best_kl = kl_val
            stagnant_iterations = 0  # Reset stagnant counter on improvement
        else:
            stagnant_iterations += 1

        # Calculate new effective percentage using sophisticated fluctuation
        old_pct = effective_pct

        # Check for forced reset condition first
        if stagnant_iterations >= max_stagnant_iterations and effective_pct > target_pct:
            effective_pct = target_pct
            stagnant_iterations = 0  # Reset counter after forced reset
            printc(f"  FORCED RESET: Been stagnant for {max_stagnant_iterations} iterations. Resetting effective_pct: {old_pct:.1f}% → {effective_pct:.1f}%", "red")
        else:
            # Apply sophisticated fluctuation strategy
            effective_pct = calculate_sophisticated_fluctuation(iteration, effective_pct, kl_val, improvement)

            # Log the fluctuation strategy and result
            if improvement > min_delta:
                status_color = "green"
                status = f"IMPROVEMENT ({fluctuation_strategy})"
            else:
                status_color = "orange" if stagnant_iterations < max_stagnant_iterations else "red"
                status = f"NO IMPROVEMENT ({fluctuation_strategy}, {stagnant_iterations}/{max_stagnant_iterations} stagnant)"

            printc(f"  {status}: effective_pct: {old_pct:.1f}% → {effective_pct:.1f}%", status_color)

        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final pruning back to target_pct
    printc(f"Final pruning: Building masks with target_pct={target_pct:.1f}%", "blue")
    final_masks = build_masks(flat_sums, target_pct)

    # Log final mask statistics
    final_selected = sum(m.count_nonzero().item() for m in final_masks.values())
    final_total = sum(m.numel() for m in final_masks.values())
    final_sparsity = (final_selected / final_total * 100) if final_total > 0 else 0
    printc(f"Final masks: {final_selected:,}/{final_total:,} params ({final_sparsity:.2f}%)", "green")

    printc(f"Evaluating final KL divergence...", "cyan")
    final_kl = eval_kl(final_masks)
    printc(f"Final KL divergence: {final_kl:.6f}", "blue")

    summary: Dict[str, Any] = {}
    for name, mask in final_masks.items():
        total = mask.numel()
        ones = int(mask.count_nonzero().item())
        density = (ones / total) if total > 0 else 0.0
        summary[name] = {
            "shape": list(shapes[name]),
            "active_count": ones,
            "total_count": total,
            "density": density,
            "sparsity": 1.0 - density,
        }

    printc(f"Generated summary for {len(summary)} parameter groups", "green")

    # Optional save
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        if save_path.endswith(".pt"):
            # Convert boolean masks to sparse COO tensors for efficient storage
            sparse_masks = convert_masks_to_sparse_coo(final_masks)
            torch.save({"masks": sparse_masks, "summary": summary, "kl_divergence": final_kl}, save_path)
        elif save_path.endswith(".json"):
            with open(save_path, "w") as f:
                json.dump({"summary": summary, "kl_divergence": final_kl}, f, indent=2)

    return {"masks": final_masks, "summary": summary, "kl_divergence": final_kl}