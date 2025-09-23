"""
Unified Dataset Builder Script

Creates a Hugging Face Dataset with fields: `prompt`, `completion`.
Merges multiple source datasets, each contributing 4096 examples.
Output: ./data/unified_prompt_completion (DatasetDict with train split)
"""

import os
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi, HfFolder

# Environment defaults
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

# Number of samples per dataset
N_SAMPLES = 4096

# Output directory
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./data/unified_prompt_completion")

# Source datasets and how to map them to prompt/completion
DATASETS = {
    "FreedomIntelligence/medical-o1-reasoning-SFT": {
        "split": "train",
        "config": "en_mix",  # must choose one: en, zh, en_mix, zh_mix
        "map_fn": lambda ex: {
            "prompt": ex.get("Question", ""),
            "completion": ex.get("Response", ""),
        },
    },
    "jtatman/python-code-dataset-500k": {
        "split": "train",
        "config": None,
        "map_fn": lambda ex: {
            "prompt": ex.get("instruction", ""),
            "completion": ex.get("output", ""),
        },
    },
    "yahma/alpaca-cleaned": {
        "split": "train",
        "config": None,
        "map_fn": lambda ex: {
            "prompt": (ex.get("instruction", "") + "\n" + ex.get("input", "")).strip(),
            "completion": ex.get("output", ""),
        },
    },
    "TIGER-Lab/MathInstruct": {
        "split": "train",
        "config": None,
        "map_fn": lambda ex: {
            "prompt": ex.get("instruction", ""),
            "completion": ex.get("output", ""),
        },
    },
}


def sample_and_map(dataset_name, cfg):
    print(f"Loading {dataset_name}...")
    if cfg["config"]:
        ds = load_dataset(dataset_name, cfg["config"], split=cfg["split"])
    else:
        ds = load_dataset(dataset_name, split=cfg["split"])

    # Sample N_SAMPLES entries
    if len(ds) > N_SAMPLES:
        ds = ds.shuffle(seed=42).select(range(N_SAMPLES))

    # Map to prompt/completion
    mapped = ds.map(cfg["map_fn"])
    final = mapped.remove_columns([c for c in mapped.column_names if c not in {"prompt", "completion"}])

    print(f"{dataset_name}: {len(final)} examples")
    return final


def main():
    subsets = []
    for name, cfg in DATASETS.items():
        subset = sample_and_map(name, cfg)
        subsets.append(subset)

    # Concatenate all datasets
    combined = Dataset.from_dict({
        "prompt": sum([s["prompt"] for s in subsets], []),
        "completion": sum([s["completion"] for s in subsets], []),
    })

    dd = DatasetDict({"train": combined})
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dd.save_to_disk(OUTPUT_DIR)

    # Upload to Hugging Face Hub
    repo_id = os.environ.get("HF_REPO_ID", "danikhan632/standard-qa")
    dd.push_to_hub(repo_id)

    print(f"\n✅ Uploaded dataset to Hugging Face: https://huggingface.co/datasets/{repo_id}")
    print("Total examples:", len(combined))


if __name__ == "__main__":
    main()
