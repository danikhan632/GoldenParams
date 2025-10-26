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
SFT Example with MaskedAdamW Optimizer

This example demonstrates how to use the SFTTrainer from the trl library to
perform supervised fine-tuning of a pre-trained language model with a
MaskedAdamW optimizer.

Dependencies:
    pip install trl transformers datasets golden_params

Example usage (with golden masks):
    python3 examples/sft_masked_example.py \
        --model_name_or_path "facebook/opt-350m" \
        --dataset_name "trl-internal-testing/sft-dialogue-demo" \
        --output_dir "sft-masked-output" \
        --golden_mask_path ./golden_masks.pt \
        --masked_lr 1e-3 \
        --unmasked_lr 1e-5 \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 2 \
        --num_train_epochs 3 \
        --max_seq_length 512
"""

import argparse
import torch
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from golden_params.optimizers.masked_adamw import MaskedAdamW
from golden_params.utils import convert_masks_to_torch_sparse_coo


@dataclass
class SFTMaskedTrainingArguments:
    """Training arguments for SFT with MaskedAdamW."""
    model_name_or_path: str = "facebook/opt-350m"
    dataset_name: str = "trl-internal-testing/sft-dialogue-demo"
    output_dir: str = "sft-masked-output"
    learning_rate: float = 1e-4  # Fallback for standard AdamW
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    num_train_epochs: int = 3
    max_seq_length: int = 512
    logging_steps: int = 10
    save_steps: int = 500
    push_to_hub: bool = False
    golden_mask_path: Optional[str] = None
    masked_lr: float = 1e-3
    unmasked_lr: float = 1e-5


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning with MaskedAdamW")
    parser.add_argument("--model_name_or_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--dataset_name", type=str, default="trl-internal-testing/sft-dialogue-demo")
    parser.add_argument("--output_dir", type=str, default="sft-masked-output")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--golden_mask_path", type=str, default=None, help="Path to golden mask .pt file")
    parser.add_argument("--masked_lr", type=float, default=1e-3, help="Learning rate for masked parameters")
    parser.add_argument("--unmasked_lr", type=float, default=1e-5, help="Learning rate for unmasked parameters")

    args = parser.parse_args()
    return SFTMaskedTrainingArguments(**vars(args))


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
    param_names = {name: param for name, param in model.named_parameters()}
    optimizer.add_param_names(param_names)
    return optimizer


def create_cosine_scheduler(optimizer, num_training_steps, num_warmup_steps=None):
    """Create a cosine annealing scheduler with warmup."""
    from transformers import get_cosine_schedule_with_warmup

    if num_warmup_steps is None:
        num_warmup_steps = num_training_steps // 10  # 10% warmup

    return get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )


def main():
    """Main training function."""
    args = parse_args()

    # 1. Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load dataset
    dataset = load_dataset(args.dataset_name, split="train")

    # 3. Configure training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        fp16=True,
        push_to_hub=args.push_to_hub,
    )

    # 4. Load golden masks and create optimizer/scheduler
    sparse_masks = None
    if args.golden_mask_path:
        print(f"Loading golden masks from: {args.golden_mask_path}")
        try:
            mask_data = torch.load(args.golden_mask_path, map_location='cpu')
            masks = mask_data.get("masks", mask_data.get("intersection_masks", {}))
            if masks:
                all_sparse = all(isinstance(m, torch.Tensor) and m.is_sparse for m in masks.values())
                if not all_sparse:
                    sparse_masks = convert_masks_to_torch_sparse_coo(masks)
                else:
                    sparse_masks = masks
                print(f"Loaded masks for {len(sparse_masks)} parameters")
        except Exception as e:
            print(f"Warning: Failed to load golden masks: {e}. Falling back to standard AdamW.")
            sparse_masks = None

    num_training_steps = (len(dataset) // (args.per_device_train_batch_size * args.gradient_accumulation_steps)) * args.num_train_epochs

    if sparse_masks:
        optimizer = create_masked_adamw_optimizer(model, sparse_masks, args.masked_lr, args.unmasked_lr)
        print(f"Created MaskedAdamW optimizer (masked_lr={args.masked_lr}, unmasked_lr={args.unmasked_lr})")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        print(f"Created standard AdamW optimizer (lr={args.learning_rate})")

    lr_scheduler = create_cosine_scheduler(optimizer, num_training_steps)
    
    # 5. Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="dialogue",
        max_seq_length=args.max_seq_length,
        packing=True,
        optimizers=(optimizer, lr_scheduler)
    )

    # 6. Start training
    trainer.train()

    # 7. Save the model
    trainer.save_model(args.output_dir)
    print(f"Model saved to {args.output_dir}")

    # 8. Push to Hub if requested
    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
