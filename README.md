# Golden Parameters

A Python package for computing golden parameters and reverse golden parameters for neural network models, featuring sophisticated fluctuation strategies and KL divergence evaluation.

## Overview

The Golden Parameters package provides functionality for analyzing parameter importance in neural networks through:

- **Golden Parameters**: Intersection-based parameter selection across multiple samples
- **Reverse Golden Parameters**: Adaptive parameter search to maximize KL divergence
- **Sophisticated Fluctuation Strategies**: Multiple algorithms for dynamic parameter exploration
- **Efficient GPU Implementation**: CUDA-optimized with sparse tensor operations

## Features

- 🧠 **Multiple Analysis Methods**: Both forward and reverse golden parameter computation
- 🚀 **GPU Accelerated**: Efficient CUDA implementations with memory optimization
- 🔬 **Advanced Algorithms**: Sophisticated fluctuation strategies including sinusoidal, momentum-based, and chaos-driven approaches
- 🔧 **Easy Integration**: Mixin class for seamless integration with existing training pipelines
- 📊 **Comprehensive Metrics**: KL divergence evaluation and detailed parameter statistics

## Installation

### From Source

```bash
git clone https://github.com/yourusername/golden_params.git
cd golden_params
pip install -e .
```

### For Development

```bash
pip install -e ".[dev]"
```

## Quick Start

### Standalone Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from golden_params import get_reverse_golden_params

# Load model and data
model = AutoModelForCausalLM.from_pretrained("your-model")
tokenizer = AutoTokenizer.from_pretrained("your-model")
dataset = load_dataset("your-dataset")

# Define prompt function
def prompt_func(samples, processing_class):
    # Your prompt processing logic
    return query_only, query_responses

# Compute reverse golden parameters
results = get_reverse_golden_params(
    model=model,
    eval_dataset=dataset,
    prompt_func=prompt_func,
    processing_class=tokenizer,
    top_k_percent=5.0,
    fluctuation_strategy="adaptive_sinusoidal"
)

print(f"KL divergence: {results['kl_divergence']}")
print(f"Generated masks for {len(results['masks'])} parameter groups")
```

### Mixin Integration

```python
from golden_params import GoldilocksMixin

class YourTrainer(GoldilocksMixin):
    def __init__(self, model, accelerator, processing_class):
        self.model = model
        self.accelerator = accelerator
        self.processing_class = processing_class

    def compute_golden_params(self, eval_dataset, prompt_func):
        return self.get_golden_params(
            eval_dataset=eval_dataset,
            prompt_func=prompt_func,
            top_k_percent=5.0
        )
```

## Fluctuation Strategies

The package supports multiple sophisticated fluctuation strategies for reverse golden parameter search:

- **`adaptive_sinusoidal`**: Sinusoidal fluctuation with adaptive amplitude
- **`momentum_based`**: Momentum-driven fluctuation with inertia
- **`exploration_decay`**: Temperature-based exploration with gradual cooling
- **`harmonic_oscillator`**: Damped harmonic oscillator dynamics
- **`chaos_driven`**: Chaotic fluctuation using logistic map

## API Reference

### Core Functions

#### `get_reverse_golden_params(model, eval_dataset, prompt_func, ...)`

Compute reverse golden parameters using adaptive search with sophisticated fluctuation strategies.

**Parameters:**
- `model`: The neural network model
- `eval_dataset`: Dataset for gradient computation and evaluation
- `prompt_func`: Function to convert dataset items to model inputs
- `top_k_percent`: Target sparsity percentage (default: 30.0)
- `fluctuation_strategy`: Strategy for parameter fluctuation (default: "adaptive_sinusoidal")
- `num_samples`: Number of dataset samples to use (default: 4)
- `eval_sample_size`: Samples for KL evaluation (default: 8)

**Returns:**
- Dictionary with `masks`, `summary`, and `kl_divergence`

### Mixin Class

#### `GoldilocksMixin`

Provides golden parameters functionality as a mixin class.

**Methods:**
- `get_golden_params()`: Compute intersection-based golden parameters
- `get_reverse_golden_params()`: Compute reverse golden parameters
- `compare_generations_with_mask()`: Compare model outputs with/without masks

## Examples

See the `example_goldy.py` file for a complete example using the standard-qa dataset.

## Requirements

- Python >= 3.8
- PyTorch >= 1.9.0
- Transformers >= 4.0.0
- Datasets >= 2.0.0
- NumPy >= 1.20.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{golden_params,
  title={Golden Parameters: A Python Package for Neural Network Parameter Importance Analysis},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/golden_params}
}
```