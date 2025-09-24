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
from golden_params.utils import printc

torch.manual_seed(420)

compute_dtype = torch.bfloat16
device = torch.device("cuda")
model_name = "Qwen/Qwen3-4B"

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

# Debug: Print dataset info
printc(f"Dataset loaded: {len(eval_dataset)} samples", "blue")
printc(f"First sample: {eval_dataset[0]}", "blue")
printc(f"Sample type: {type(eval_dataset[0])}", "blue")

def prompt_func(samples, processing_class):
    """Simple prompt function for standard-qa dataset"""
    import random

    # Handle different dataset structures
    questions = []
    answers = []

    # Add some variety to answers to prevent identical prompts
    answer_options = ["Yes", "No", "Maybe", "True", "False", "Correct", "Possible"]

    for sample in samples:
        if isinstance(sample, dict):
            # If sample is a dictionary, extract question and answer
            if "question" in sample and "answer" in sample:
                questions.append(sample["question"])
                answers.append(sample["answer"])
            elif "text" in sample:
                # If it's a text field, try to parse it
                text = sample["text"]
                questions.append(text[:100])  # Use first 100 chars as question
                answers.append(random.choice(answer_options))  # Random answer for variety
            else:
                # Fallback for unknown dict structure
                questions.append(str(sample))
                answers.append(random.choice(answer_options))
        elif isinstance(sample, str):
            # If sample is a string, use it as question
            questions.append(sample[:100])  # Truncate to reasonable length
            answers.append(random.choice(answer_options))  # Random answer for variety
        else:
            # Fallback for any other type
            questions.append(str(sample))
            answers.append(random.choice(answer_options))

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
        num_samples=2048,
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