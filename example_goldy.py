"""
Golden Mask Generation Script (Using golden_params package)

- Loads and cleans the `danikhan632/standard-qa` dataset.
- Uses a simplified prompt function for prompt/completion pairs.
- Generates reverse golden masks using the golden_params package.
"""
import os
# Avoid tokenizers fork deadlocks with multiprocessing/dataloaders
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from golden_params import get_reverse_golden_params
from utils import printc

torch.manual_seed(420)

compute_dtype = torch.bfloat16
device = torch.device("cuda")
model_name = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=compute_dtype,
    trust_remote_code=True,
)
model = model.to(device)

# Load dataset
dataset = load_dataset("danikhan632/standard-qa", split="train")
eval_dataset = dataset.select(range(min(50, len(dataset))))

def prompt_func(samples, processing_class):
    """Simple prompt function for standard-qa dataset"""
    questions = [sample["question"] for sample in samples]
    answers = [sample["answer"] for sample in samples]

    # Create prompt/completion pairs
    prompts = [f"Question: {q}\nAnswer: {a}" for q, a in zip(questions, answers)]

    # Tokenize
    inputs = processing_class(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    # Query only (just the questions)
    query_prompts = [f"Question: {q}\nAnswer:" for q in questions]
    query_only = processing_class(
        query_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    )

    return query_only["input_ids"].to(device), inputs["input_ids"].to(device)

# Main execution
if __name__ == "__main__":
    printc("Starting reverse golden parameters generation...", "blue")

    # Run reverse golden parameter search with sophisticated fluctuation
    results = get_reverse_golden_params(
        model=model,
        eval_dataset=eval_dataset,
        prompt_func=prompt_func,
        processing_class=tokenizer,
        top_k_percent=5.0,
        num_samples=16,
        eval_sample_size=8,
        chunk_size=4,
        batch_size=1,
        fluctuation_strategy="adaptive_sinusoidal",  # Try: "momentum_based", "exploration_decay", "harmonic_oscillator", "chaos_driven"
        fluctuation_amplitude=0.4,
        fluctuation_frequency=0.15,
        momentum_factor=0.85,
        exploration_temperature=1.2,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        save_path="reverse_golden_masks.pt"
    )

    printc(f"Reverse golden parameters generation complete!", "green")
    printc(f"Final KL divergence: {results['kl_divergence']:.6f}", "blue")
    printc(f"Generated masks for {len(results['masks'])} parameter groups", "cyan")
    printc("Results saved to: reverse_golden_masks.pt", "yellow")