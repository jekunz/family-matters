"""
Corpus of the joint multilingual baseline: sv, da, nb (4B tokens each), all
nn and is data, shuffled (train.py's --max_tokens keeps the first chunks).

Usage:
    python preprocess/preprocess_multilingual_baseline.py --output_dir data/multilingual_baseline
"""

import argparse
import logging
import os

from datasets import DatasetDict, concatenate_datasets
from transformers import AutoTokenizer

from data_loader import MAX_SEQ_LENGTH, tokenize_fineweb

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

# (FineWeb-2 config, documents to tokenize, token budget); None = everything
LANGUAGES = [
    ("swe_Latn", 10_000_000, 4_000_000_000),
    ("dan_Latn", 10_000_000, 4_000_000_000),
    ("nob_Latn", 10_000_000, 4_000_000_000),
    ("nno_Latn", None, None),
    ("isl_Latn", None, None),
]
SHUFFLE_SEED = 42


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    p.add_argument("--num_workers", type=int, default=12)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    train_parts, val_parts = [], []
    for config, max_samples, max_tokens in LANGUAGES:
        dataset = tokenize_fineweb(tokenizer, config, max_samples, args.num_workers)
        if max_tokens:
            dataset["train"] = dataset["train"].select(range(min(max_tokens // MAX_SEQ_LENGTH, len(dataset["train"]))))
        print(f"{config}: {len(dataset['train']):,} chunks")
        train_parts.append(dataset["train"])
        val_parts.append(dataset["validation"])

    combined = DatasetDict({
        "train": concatenate_datasets(train_parts).shuffle(seed=SHUFFLE_SEED),
        "validation": concatenate_datasets(val_parts).shuffle(seed=SHUFFLE_SEED),
    })
    combined.save_to_disk(args.output_dir)
    tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer"))
    print(f"Saved {len(combined['train']):,} training chunks to {args.output_dir}")


if __name__ == "__main__":
    main()
