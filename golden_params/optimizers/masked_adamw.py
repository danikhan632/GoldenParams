# mypy: allow-untyped-defs
from typing import Optional, Union, Dict, List
import warnings

import torch
from torch import Tensor
from torch.optim import AdamW

from ..sparse_mask_utils import SparseMaskManager


__all__ = ["MaskedAdamW"]


class MaskedAdamW(AdamW):
    def __init__(
        self,
        params,
        sparse_masks: Dict[str, torch.Tensor],
        masked_lr: Union[float, Tensor] = 1e-3,
        unmasked_lr: Union[float, Tensor] = 1e-4,
        betas: tuple[Union[float, Tensor], Union[float, Tensor]] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        # Validate sparse masks
        if not isinstance(sparse_masks, dict):
            raise TypeError("sparse_masks must be a dictionary")

        for name, mask in sparse_masks.items():
            if not mask.is_sparse:
                raise ValueError(f"Mask for parameter '{name}' must be a sparse COO tensor")
            if not mask.is_coalesced():
                sparse_masks[name] = mask.coalesce()

        self.sparse_masks = sparse_masks
        self.masked_lr = masked_lr
        self.unmasked_lr = unmasked_lr

        # Parameter name mapping for quick lookup
        self._param_name_map = {}

        # Track skipped masks (for mismatched shapes)
        self._skipped_masks = {}  # param_name -> (mask_shape, param_shape)
        self._warned_once = False

        # Initialize with unmasked learning rate as default
        super().__init__(
            params,
            lr=unmasked_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            foreach=foreach,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )

        # Store additional parameters in defaults
        for group in self.param_groups:
            group['masked_lr'] = masked_lr
            group['unmasked_lr'] = unmasked_lr
            group['sparse_masks'] = sparse_masks

    def _build_param_name_map(self):
        """Build mapping from parameter tensors to their names."""
        if self._param_name_map:
            return

        for group in self.param_groups:
            if 'param_names' in group:
                for param, name in zip(group['params'], group['param_names']):
                    self._param_name_map[id(param)] = name

        # If no explicit names provided, warn user
        if not self._param_name_map:
            warnings.warn(
                "No parameter names provided. Cannot apply sparse masks. "
                "Pass parameter names via param_groups or use add_param_names method.",
                UserWarning
            )

    def add_param_names(self, param_names: Dict[str, torch.nn.Parameter]):
        """
        Add parameter name mapping.

        Args:
            param_names: Dictionary mapping parameter names to parameter tensors
        """
        for name, param in param_names.items():
            self._param_name_map[id(param)] = name

    def step(self, closure=None):
        """Perform a single optimization step with masked learning rates."""
        self._cuda_graph_capture_health_check()
        self._build_param_name_map()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # Separate parameters into masked and unmasked groups
            masked_params = []
            unmasked_params = []
            masked_grads = []
            unmasked_grads = []

            # State tracking for each group
            masked_exp_avgs = []
            unmasked_exp_avgs = []
            masked_exp_avg_sqs = []
            unmasked_exp_avg_sqs = []
            masked_max_exp_avg_sqs = []
            unmasked_max_exp_avg_sqs = []
            masked_state_steps = []
            unmasked_state_steps = []

            has_complex = False

            for param in group['params']:
                if param.grad is None:
                    continue

                has_complex |= torch.is_complex(param)

                if param.grad.is_sparse:
                    raise RuntimeError(
                        "MaskedAdamW does not support sparse gradients"
                    )

                # Get parameter name
                param_name = self._param_name_map.get(id(param))

                # Check if this parameter has a sparse mask
                if param_name and param_name in self.sparse_masks:
                    sparse_mask = self.sparse_masks[param_name]

                    # Validate mask shape matches parameter shape
                    if sparse_mask.shape != param.shape:
                        # Track skipped mask with shape info
                        if param_name not in self._skipped_masks:
                            self._skipped_masks[param_name] = (sparse_mask.shape, param.shape)

                        # Treat as unmasked parameter
                        sparse_mask = None

                    if sparse_mask is not None:
                        # Apply sparse mask to create masked and unmasked updates
                        masked_grad, unmasked_grad = self._split_gradient(param.grad, sparse_mask)

                        if masked_grad is not None:
                            masked_params.append(param)
                            masked_grads.append(masked_grad)

                            # Initialize state if needed
                            state = self.state[param]
                            if len(state) == 0:
                                self._init_param_state(param, group)

                            masked_exp_avgs.append(state["exp_avg"])
                            masked_exp_avg_sqs.append(state["exp_avg_sq"])
                            if group["amsgrad"]:
                                masked_max_exp_avg_sqs.append(state["max_exp_avg_sq"])
                            masked_state_steps.append(state["step"])

                        if unmasked_grad is not None:
                            unmasked_params.append(param)
                            unmasked_grads.append(unmasked_grad)

                            # Use same state objects (they'll be updated with combined effects)
                            state = self.state[param]
                            if len(state) == 0:
                                self._init_param_state(param, group)

                            unmasked_exp_avgs.append(state["exp_avg"])
                            unmasked_exp_avg_sqs.append(state["exp_avg_sq"])
                            if group["amsgrad"]:
                                unmasked_max_exp_avg_sqs.append(state["max_exp_avg_sq"])
                            unmasked_state_steps.append(state["step"])
                    else:
                        # No valid mask - treat as unmasked
                        unmasked_params.append(param)
                        unmasked_grads.append(param.grad)

                        state = self.state[param]
                        if len(state) == 0:
                            self._init_param_state(param, group)

                        unmasked_exp_avgs.append(state["exp_avg"])
                        unmasked_exp_avg_sqs.append(state["exp_avg_sq"])
                        if group["amsgrad"]:
                            unmasked_max_exp_avg_sqs.append(state["max_exp_avg_sq"])
                        unmasked_state_steps.append(state["step"])
                else:
                    # No mask for this parameter, treat as unmasked
                    unmasked_params.append(param)
                    unmasked_grads.append(param.grad)

                    state = self.state[param]
                    if len(state) == 0:
                        self._init_param_state(param, group)

                    unmasked_exp_avgs.append(state["exp_avg"])
                    unmasked_exp_avg_sqs.append(state["exp_avg_sq"])
                    if group["amsgrad"]:
                        unmasked_max_exp_avg_sqs.append(state["max_exp_avg_sq"])
                    unmasked_state_steps.append(state["step"])

            # Apply updates manually using simplified AdamW logic
            self._apply_updates(masked_params, masked_grads, group["masked_lr"], group)
            self._apply_updates(unmasked_params, unmasked_grads, group["unmasked_lr"], group)

        # Print mismatch summary once after first step
        if self._skipped_masks and not self._warned_once:
            self._warned_once = True
            print("\n" + "=" * 70)
            print(f"⚠ MaskedAdamW: {len(self._skipped_masks)} masks skipped due to shape mismatch")
            print("=" * 70)
            print(f"{'Parameter Name':<50} {'Mask Shape':<20} {'Param Shape':<20}")
            print("-" * 70)
            for param_name, (mask_shape, param_shape) in sorted(self._skipped_masks.items()):
                print(f"{param_name:<50} {str(mask_shape):<20} {str(param_shape):<20}")
            print("-" * 70)
            print(f"These {len(self._skipped_masks)} parameters will use unmasked_lr={self.unmasked_lr}")
            print("=" * 70 + "\n")

        return loss

    def _init_param_state(self, param, group):
        """Initialize parameter state."""
        state = self.state[param]

        # Initialize step counter - use float for simplicity
        state["step"] = torch.tensor(0.0)

        # Initialize momentum and variance estimates
        state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
        state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)

        if group["amsgrad"]:
            state["max_exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)

    def _apply_updates(self, params, grads, lr, group):
        """Apply AdamW updates to a set of parameters with given learning rate."""
        if not params:
            return

        for param, grad in zip(params, grads):
            if grad is None:
                continue

            state = self.state[param]

            # Get state variables
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            step_t = state["step"]

            # Increment step
            step_t += 1

            beta1, beta2 = group["betas"]

            # Apply weight decay (decoupled) - use data to avoid gradient tracking
            param.data.mul_(1 - lr * group["weight_decay"])

            # Exponential moving average of gradient values
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

            # Exponential moving average of squared gradient values
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            # Bias correction
            step = step_t.item()
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            step_size = lr / bias_correction1

            if group["amsgrad"]:
                # Maintains the maximum of all 2nd moment running avg. till now
                max_exp_avg_sq = state["max_exp_avg_sq"]
                torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group["eps"])
            else:
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group["eps"])

            # Apply update - use data to avoid gradient tracking
            if group["maximize"]:
                param.data.addcdiv_(exp_avg, denom, value=step_size)
            else:
                param.data.addcdiv_(exp_avg, denom, value=-step_size)

    def _split_gradient(self, grad, sparse_mask):
        """
        Split gradient into masked and unmasked components.

        Args:
            grad: Full gradient tensor
            sparse_mask: Sparse COO mask tensor

        Returns:
            Tuple of (masked_grad, unmasked_grad) where each is None if no updates needed
        """
        # Get dense mask for efficient operations
        dense_mask = sparse_mask.to_dense().bool()

        # Create masked gradient (only update positions marked True in mask)
        masked_grad = None
        if dense_mask.any():
            masked_grad = torch.zeros_like(grad)
            masked_grad[dense_mask] = grad[dense_mask]

        # Create unmasked gradient (update positions marked False in mask)
        unmasked_grad = None
        unmasked_mask = ~dense_mask
        if unmasked_mask.any():
            unmasked_grad = torch.zeros_like(grad)
            unmasked_grad[unmasked_mask] = grad[unmasked_mask]

        return masked_grad, unmasked_grad

    def get_lr_info(self):
        """Get information about current learning rates."""
        info = {
            "masked_lr": self.masked_lr,
            "unmasked_lr": self.unmasked_lr,
            "masks_loaded": len(self.sparse_masks),
            "param_name_mapping": len(self._param_name_map)
        }

        # Calculate mask statistics
        total_masked_params = 0
        total_unmasked_params = 0

        for mask in self.sparse_masks.values():
            density = SparseMaskManager.sparse_mask_density(mask)
            total_elements = mask.numel()
            masked_elements = int(density * total_elements)
            unmasked_elements = total_elements - masked_elements

            total_masked_params += masked_elements
            total_unmasked_params += unmasked_elements

        info["total_masked_parameters"] = total_masked_params
        info["total_unmasked_parameters"] = total_unmasked_params
        info["skipped_masks"] = len(self._skipped_masks)
        info["skipped_mask_names"] = list(self._skipped_masks.keys()) if self._skipped_masks else []
        info["skipped_mask_details"] = self._skipped_masks  # dict of param_name -> (mask_shape, param_shape)

        return info




MaskedAdamW.__doc__ = (
    r"""Implements Masked AdamW algorithm with separate learning rates for masked and unmasked parameters.

    This optimizer extends AdamW to support sparse COO tensor masks, allowing different learning rates
    to be applied to different parameter regions. Parameters marked True in the sparse mask use the
    masked_lr, while parameters marked False use the unmasked_lr.

    The algorithm maintains a single set of momentum and variance estimates per parameter, but applies
    gradient updates with different learning rates based on the sparse mask pattern.

    Key features:
    - Efficient handling of sparse masks using COO tensor format
    - Single optimizer instance (no duplication of state)
    - Memory-efficient mask application
    - Compatible with existing AdamW features (weight decay, amsgrad, etc.)

    Args:
        params: Parameters to optimize
        sparse_masks: Dictionary mapping parameter names to sparse COO mask tensors
        masked_lr: Learning rate for parameters marked True in mask (default: 1e-3)
        unmasked_lr: Learning rate for parameters marked False in mask (default: 1e-4)
        betas: Coefficients for computing running averages (default: (0.9, 0.999))
        eps: Term added to denominator for numerical stability (default: 1e-8)
        weight_decay: Weight decay coefficient (default: 1e-2)
        amsgrad: Whether to use AMSGrad variant (default: False)
        maximize: Whether to maximize the objective (default: False)
        foreach: Whether to use multi-tensor operations (default: None)
        capturable: Whether the optimizer is capturable (default: False)
        differentiable: Whether the optimizer is differentiable (default: False)
        fused: Whether to use fused implementation (default: None)

    Example:
        >>> import torch
        >>> from golden_params.optimizers import MaskedAdamW
        >>> from golden_params.utils import convert_masks_to_sparse_coo
        >>>
        >>> # Create model and masks
        >>> model = torch.nn.Linear(10, 1)
        >>> masks = {"weight": torch.rand(1, 10) > 0.5, "bias": torch.rand(1) > 0.5}
        >>> sparse_masks = convert_masks_to_sparse_coo(masks)
        >>>
        >>> # Create optimizer
        >>> optimizer = MaskedAdamW(
        ...     model.parameters(),
        ...     sparse_masks=sparse_masks,
        ...     masked_lr=1e-3,    # Higher LR for masked parameters
        ...     unmasked_lr=1e-4   # Lower LR for unmasked parameters
        ... )
        >>>
        >>> # Add parameter name mapping
        >>> optimizer.add_param_names({"weight": model.weight, "bias": model.bias})
        >>>
        >>> # Standard training loop
        >>> loss = model(torch.randn(5, 10)).sum()
        >>> loss.backward()
        >>> optimizer.step()
        >>> optimizer.zero_grad()

    Note:
        Parameter names must be provided either via param_groups or using add_param_names()
        to enable mask application. Without names, the optimizer will function as regular AdamW
        with unmasked_lr for all parameters.
    """
)