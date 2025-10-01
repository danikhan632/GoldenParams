# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
GRPO Example with MaskedAdamW Optimizer

This example demonstrates how to use GRPO (Group Relative Policy Optimization)
with the MaskedAdamW optimizer from golden_params package for efficient sparse
mask training. The optimizer applies different learning rates to masked vs
unmasked parameters based on a golden mask.

Dependencies:
    pip install trl math_verify latex2sympy2_extended trackio kernels

For Qwen models:
    pip install num2words==0.5.14

Example usage (without golden masks - uses standard AdamW):
    python3 examples/grpo_msked_example.py \
        --model_name_or_path Shekswess/trlm-135m \
        --output_dir grpo-output \
        --learning_rate 1e-5 \
        --max_prompt_length 2048 \
        --max_completion_length 1024 \
        --log_completions \
        --per_device_train_batch_size 8 \
        --num_generations 8 \
        --importance_sampling_level sequence \
        --epsilon 3e-4 \
        --epsilon_high 4e-4 \
        --beta 0.0 \
        --loss_type grpo \
        --gradient_accumulation_steps 2 \
        --steps_per_generation 8

Example usage (with golden masks - uses MaskedAdamW):
    python3 examples/grpo_msked_example.py \
        --model_name_or_path Shekswess/trlm-135m \
        --output_dir grpo-output \
        --golden_mask_path ./golden_masks.pt \
        --masked_lr 1e-3 \
        --unmasked_lr 1e-5 \
        --max_prompt_length 2048 \
        --max_completion_length 1024 \
        --log_completions \
        --per_device_train_batch_size 8 \
        --num_generations 8 \
        --importance_sampling_level sequence \
        --epsilon 3e-4 \
        --epsilon_high 4e-4 \
        --beta 0.0 \
        --loss_type grpo \
        --gradient_accumulation_steps 2 \
        --steps_per_generation 8
"""

import os
import torch
import argparse
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset

from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from golden_params.optimizers.masked_adamw import MaskedAdamW
from golden_params.utils import convert_masks_to_torch_sparse_coo

# # Enable logging in a Hugging Face Space
# os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")
from openai import OpenAI
import json
import os
# Get the API key from environment variable
api_key = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="http://127.0.0.1:8040/v1",
    api_key="",
)

def extract_context(text: str):
    marker = "</think>"
    idx = text.find(marker)
    if idx == -1:
        return None, None  # no closing tag found
    
    # 100 characters before </think>
    before = text[max(0, idx-100):idx]
    
    # everything after </think>
    after = text[idx+len(marker):]
    
    return before +"\n"+after




def check_answer(question: str, potential_solution: str, correct_solution: str):
    completion = client.chat.completions.create(
        model="alpindale/Llama-3.2-1B-Instruct",
        messages=[
            {"role": "system", "content": "You are a strict answer checker. Only check correctness. If the reponse contains non-sense or irrelevant content then mark it as noisy"},
            {"role": "user", "content": f"Question: {question}\n STUDENT SOLUTION: {extract_context(potential_solution)}\nACTUAL CORRECT SOLUTION: {correct_solution}"}
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "evaluate_answer",
                    "description": "Check if the given answer matches the correct solution",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "is_correct": {
                                "type": "boolean",
                                "description": "True if the potential solution is correct, otherwise False"
                            },
                            "is_noisy": {
                                "type": "boolean",
                                "description": "True if the potential solution contains non-sense or irrelevant content, otherwise False"
                            },
                        },
                        "required": ["is_correct", "is_noisy"]
                    }
                }
            }
        ],
        tool_choice={"type": "function", "function": {"name": "evaluate_answer"}},
    )
    print(extract_context(potential_solution))
    msg = completion.choices[0].message.content
    # print(completion.choices[0])
    

    # Extract function call arguments
    tool_calls = completion.choices[0].message.tool_calls
    if tool_calls is None and msg is not None:
        args = json.loads(msg)
    else:
        args = json.loads(tool_calls[0].function.arguments)
    print(args)
    score = 0.0
    if args['is_correct']:
        score+=0.5
    return score
    
    


@dataclass
class TrainingArguments:
    """Simple training arguments to avoid TRL parser issues."""
    model_name_or_path: str = "Shekswess/trlm-135m"
    output_dir: str = "grpo-output"
    learning_rate: float = 1e-5
    dtype: str = "bfloat16"
    max_prompt_length: int = 2048
    max_completion_length: int = 1024
    log_completions: bool = False
    per_device_train_batch_size: int = 8
    num_generations: int = 8
    importance_sampling_level: str = "sequence"
    epsilon: float = 3e-4
    epsilon_high: float = 4e-4
    beta: float = 0.0
    loss_type: str = "grpo"
    gradient_accumulation_steps: int = 2
    steps_per_generation: int = 4
    num_train_epochs: int = 1
    max_steps: int = -1
    save_steps: int = 500
    logging_steps: int = 10
    push_to_hub: bool = False
    dataset_name: Optional[str] = None
    golden_mask_path: Optional[str] = None
    masked_lr: float = 1e-3
    unmasked_lr: float = 1e-5

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GRPO Training with Golden Parameters")
    parser.add_argument("--model_name_or_path", type=str, default="Shekswess/trlm-135m")
    parser.add_argument("--output_dir", type=str, default="grpo-output")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_completion_length", type=int, default=512)
    parser.add_argument("--log_completions", action="store_true")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--importance_sampling_level", type=str, default="sequence")
    parser.add_argument("--epsilon", type=float, default=3e-4)
    parser.add_argument("--epsilon_high", type=float, default=4e-4)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--loss_type", type=str, default="grpo")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--steps_per_generation", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--golden_mask_path", type=str, default=None, help="Path to golden mask .pt file")
    parser.add_argument("--masked_lr", type=float, default=1e-3, help="Learning rate for masked parameters")
    parser.add_argument("--unmasked_lr", type=float, default=1e-5, help="Learning rate for unmasked parameters")

    args = parser.parse_args()
    return TrainingArguments(**vars(args))

def load_and_prepare_dataset():
    """Load and prepare the GSM8K dataset."""
    dataset = load_dataset("reasoning-machines/gsm-hard", split="train")

    # 90/10 split
    splits = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = splits["train"]
    eval_dataset = splits["test"]

    def make_conversation(example):
        return {
            "prompt": [
                {"role": "user", "content": example["input"]},
            ],
            "answer": str(example["target"]) +"\nlogic for answer\n"+ example["code"] 
        }

    train_dataset = train_dataset.map(make_conversation)
    eval_dataset = eval_dataset.map(make_conversation)



    return train_dataset, eval_dataset

from concurrent.futures import ThreadPoolExecutor

def accuracy_reward(completions, answer: list[str], **kwargs):
    """
    Reward function that uses OpenAI API to check if completions are correct.

    Args:
        completions: List of completion dictionaries
        answer: List of reference answers
        **kwargs: Additional arguments

    Returns:
        List of reward scores (float)
    """
    contents = [completion[0]["content"] for completion in completions]

    # Extract question from prompts if available
    prompts = kwargs.get('prompts', [])
    question = ""
    if prompts and len(prompts) > 0 and len(prompts[0]) > 1:
        question = prompts[0][1]['content']

    def evaluate(content, reference_answer):
        return check_answer(
            question=str(question),
            potential_solution=content,
            correct_solution=reference_answer
        )
        
    # Run in parallel
    with ThreadPoolExecutor() as executor:
        rewards = list(executor.map(evaluate, contents, answer))

    return rewards


def create_grpo_config(args):
    """Create GRPO configuration from training arguments."""
    return GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        importance_sampling_level=args.importance_sampling_level,
        epsilon=args.epsilon,
        epsilon_high=args.epsilon_high,
        beta=args.beta,
        loss_type=args.loss_type,
        steps_per_generation=args.steps_per_generation,
        log_completions=args.log_completions,
        push_to_hub=args.push_to_hub,
        bf16=args.dtype == "bfloat16",
        fp16=args.dtype == "float16",
        use_vllm=True,
        vllm_mode="colocate",
        # optional tuning parameters:
        vllm_gpu_memory_utilization=0.05,  # fraction of GPU memory vLLM may use

    )

def create_masked_adamw_optimizer(model, sparse_masks, masked_lr, unmasked_lr):
    """Create MaskedAdamW optimizer with sparse masks."""
    optimizer = MaskedAdamW(
        model.parameters(),
        sparse_masks=sparse_masks,
        masked_lr=masked_lr,
        unmasked_lr=unmasked_lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Add parameter name mapping
    param_names = {}
    for name, param in model.named_parameters():
        param_names[name] = param
    optimizer.add_param_names(param_names)

    return optimizer

def create_cosine_scheduler(optimizer, num_training_steps, num_warmup_steps=None):
    """Create a cosine annealing scheduler with warmup."""
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from transformers import get_cosine_schedule_with_warmup

    if num_warmup_steps is None:
        num_warmup_steps = num_training_steps // 10  # 10% warmup

    return get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

def create_trainer(args, train_dataset, eval_dataset):
    """Create and configure the GRPO trainer."""
    grpo_config = create_grpo_config(args)

    trainer = GRPOTrainer(
        model=args.model_name_or_path,
        args=grpo_config,
        reward_funcs=[accuracy_reward],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Load golden masks if provided
    sparse_masks = None
    if args.golden_mask_path:
        print(f"Loading golden masks from: {args.golden_mask_path}")
        try:
            mask_data = torch.load(args.golden_mask_path, map_location='cpu')
            # Check if masks are already sparse or need conversion
            masks = mask_data.get("masks", mask_data.get("intersection_masks", {}))

            # Convert to sparse if they're not already
            if masks:
                # Check if all values are sparse COO tensors
                all_sparse = all(
                    isinstance(m, torch.Tensor) and m.is_sparse
                    for m in masks.values()
                )

                if not all_sparse:
                    # Need to convert from dense boolean masks to sparse COO tensors
                    sparse_masks = convert_masks_to_torch_sparse_coo(masks)
                else:
                    sparse_masks = masks

            print(f"Loaded masks for {len(sparse_masks)} parameters")

            # Print mask statistics
            total_params = 0
            masked_params = 0
            for mask in sparse_masks.values():
                if mask.is_sparse:
                    mask_dense = mask.to_dense()
                    total_params += mask_dense.numel()
                    masked_params += mask_dense.count_nonzero().item()

            if total_params > 0:
                sparsity = (masked_params / total_params * 100)
                print(f"Mask statistics: {masked_params:,}/{total_params:,} params ({sparsity:.2f}% sparse)")
        except Exception as e:
            print(f"Warning: Failed to load golden masks: {e}")
            print("Falling back to standard AdamW optimizer")
            sparse_masks = None

    # Override the optimizer creation method
    def create_optimizer_and_scheduler(num_training_steps):
        if sparse_masks is not None:
            # Create MaskedAdamW optimizer
            trainer.optimizer = create_masked_adamw_optimizer(
                trainer.model,
                sparse_masks,
                args.masked_lr,
                args.unmasked_lr
            )
            print(f"Created MaskedAdamW optimizer (masked_lr={args.masked_lr}, unmasked_lr={args.unmasked_lr})")

            # Print optimizer info
            lr_info = trainer.optimizer.get_lr_info()
            print(f"  Masks loaded: {lr_info['masks_loaded']}")
            print(f"  Parameters mapped: {lr_info['param_name_mapping']}")
            if lr_info['skipped_masks'] > 0:
                print(f"  ⚠ Skipped masks (shape mismatch): {lr_info['skipped_masks']}")
                print(f"    (These parameters will use unmasked_lr={args.unmasked_lr})")
        else:
            # Fallback to standard AdamW
            trainer.optimizer = torch.optim.AdamW(
                trainer.model.parameters(),
                lr=args.learning_rate,
                weight_decay=0.01,
                betas=(0.9, 0.999),
                eps=1e-8
            )
            print(f"Created standard AdamW optimizer (lr={args.learning_rate})")

        # Create cosine scheduler
        trainer.lr_scheduler = create_cosine_scheduler(trainer.optimizer, num_training_steps)

    trainer.create_optimizer_and_scheduler = create_optimizer_and_scheduler

    return trainer

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Load and prepare dataset
    train_dataset, eval_dataset = load_and_prepare_dataset()

    # Create trainer
    trainer = create_trainer(args, train_dataset, eval_dataset)

    # Train the model
    trainer.train()

    # Save model
    trainer.save_model(args.output_dir)

    # Push to hub if requested
    if args.push_to_hub:
        trainer.push_to_hub(dataset_name=args.dataset_name)

if __name__ == "__main__":
    main()