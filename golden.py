# Standard Library
import json
import math
import os
import random
from itertools import islice
from typing import Any, Callable, Dict, List, Optional, Union
from contextlib import contextmanager, nullcontext
import copy

# Third-Party Libraries
import torch
from transformers import (
    GenerationConfig,
    PreTrainedTokenizerBase,
    TrainingArguments,
)

from golden_params.utils import (
    printc,
    forward_pass_logprobs_for_fixed_ids,
    get_kl_divergence,
)
from golden_params.sparse_mask_utils import SparseMaskManager

# Import the new golden_params package
from golden_params import GoldilocksMixin

# Re-export the mixin for backward compatibility
__all__ = ["GoldilocksMixin"]