"""
Pre-tokenize one FineWeb-2 language into 2048-token chunks for train.py.

Usage:
    python preprocess/preprocess_data.py --dataset_config fao_Latn --output_dir data/fao
"""

import argparse
import logging
import os

from transformers import AutoTokenizer

from data_loader import tokenize_fineweb

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", default="HuggingFaceTB/SmolLM2-135M-Instruct", help="Shared by both model sizes")
    p.add_argument("--dataset_config", required=True, help="FineWeb-2 config, e.g. fao_Latn")
    p.add_argument("--max_samples", type=int, default=None, help="Use only the first N documents")
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    dataset = tokenize_fineweb(tokenizer, args.dataset_config, args.max_samples, args.num_workers)
    dataset.save_to_disk(args.output_dir)
    tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer"))
    print(f"Saved {len(dataset['train']):,} training chunks to {args.output_dir}")


if __name__ == "__main__":
    main()
