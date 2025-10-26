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
SFT Example

This example demonstrates how to use the SFTTrainer from the trl library to
perform supervised fine-tuning of a pre-trained language model.

Dependencies:
    pip install trl transformers datasets

Example usage:
    python3 examples/sft_example.py \
        --model_name_or_path "facebook/opt-350m" \
        --dataset_name "trl-internal-testing/sft-dialogue-demo" \
        --output_dir "sft-output" \
        --learning_rate 1e-4 \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 2 \
        --num_train_epochs 3 \
        --max_seq_length 512
"""

import argparse
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


@dataclass
class SFTTrainingArguments:
    """Simplified training arguments for SFT."""
    model_name_or_path: str = "facebook/opt-350m"
    dataset_name: str = "trl-internal-testing/sft-dialogue-demo"
    output_dir: str = "sft-output"
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 2
    num_train_epochs: int = 3
    max_seq_length: int = 512
    logging_steps: int = 10
    save_steps: int = 500
    push_to_hub: bool = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning with TRL")
    parser.add_argument("--model_name_or_path", type=str, default="facebook/opt-350m")
    parser.add_argument("--dataset_name", type=str, default="trl-internal-testing/sft-dialogue-demo")
    parser.add_argument("--output_dir", type=str, default="sft-output")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--push_to_hub", action="store_true")

    args = parser.parse_args()
    return SFTTrainingArguments(**vars(args))


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
        fp16=True,  # Use mixed precision for efficiency
        push_to_hub=args.push_to_hub,
    )

    # 4. Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        dataset_text_field="dialogue",  # Specify the text field in the dataset
        max_seq_length=args.max_seq_length,
        packing=True,  # Pack multiple short examples into one sequence for efficiency
    )

    # 5. Start training
    trainer.train()

    # 6. Save the model
    trainer.save_model(args.output_dir)
    print(f"Model saved to {args.output_dir}")

    # 7. Push to Hub if requested
    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
