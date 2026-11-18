"""
FoBLiMP / FoBCoMP: a pair is correct if the correct text has the lower
per-token perplexity. For LoRA checkpoints, pass --base_model and --adapter_path.

Usage:
    python evaluate/eval_minimal_pairs.py --dataset FoBLiMP --model_name_or_path path/to/checkpoint \\
        --tokenizer_name HuggingFaceTB/SmolLM2-360M-Instruct \\
        --config_name individual-icelandic-360m-full --seed 1 --output_dir results/foblimp
"""

import argparse
import json
import os

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

FOBLIMP_SPLITS = ["multiblimp", "scala_delete", "scala_flip", "germdetect", "translation_pairs"]
FOBCOMP_SPLITS = ["sentiment_sentences", "sentiment_articles", "topic_classification", "qa_shuffled", "qa_generated"]

BENCHMARK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmarks")

MAX_LENGTH = 8192

# Detects a mis-loaded tokenizer (see --tokenizer_name).
TOKENIZER_PROBE = "Men hon segði: \"Ja, fyri hetta virðið.\""


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=["FoBLiMP", "FoBCoMP"])
    p.add_argument("--model_name_or_path", default=None, help="Full model checkpoint")
    p.add_argument("--base_model", default=None, help="Base model, when loading a LoRA adapter")
    p.add_argument("--adapter_path", default=None, help="LoRA adapter checkpoint")
    p.add_argument("--tokenizer_name", default=None,
                   help="Needed for checkpoints saved without tokenizer files")
    p.add_argument("--config_name", required=True, help="Label of this configuration, e.g. individual-icelandic-360m-full")
    p.add_argument("--seed", type=int, required=True, help="Training seed of this checkpoint")
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()
    if not args.model_name_or_path and not (args.base_model and args.adapter_path):
        p.error("pass either --model_name_or_path, or both --base_model and --adapter_path")
    return args


def score_text(model, tokenizer, text, device):
    """(mean per-token NLL, number of input tokens) for `text`."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH).to(device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return outputs.loss.item(), inputs["input_ids"].size(1)


def score_pair(model, tokenizer, correct_text, wrong_text, device):
    loss_c, n_c = score_text(model, tokenizer, correct_text, device)
    loss_w, n_w = score_text(model, tokenizer, wrong_text, device)
    return {"loss_correct": loss_c, "loss_wrong": loss_w, "ntok_correct": n_c, "ntok_wrong": n_w,
            "correct": loss_c < loss_w}


def load_split(dataset, split):
    """One split (benchmarks/<dataset>/<split>.parquet) with correct_text / wrong_text columns."""
    return load_dataset("parquet", data_files=os.path.join(BENCHMARK_DIR, dataset, f"{split}.parquet"), split="train")


def load_tokenizer(source):
    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    n = len(tokenizer(TOKENIZER_PROBE)["input_ids"])
    if n < 6:
        raise RuntimeError(f"tokenizer loaded from {source!r} encodes a 9-word sentence into {n} tokens -- "
                           "pass --tokenizer_name HuggingFaceTB/SmolLM2-135M-Instruct")
    return tokenizer


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = load_tokenizer(args.tokenizer_name or args.base_model or args.model_name_or_path)
    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    else:
        model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(args.base_model), args.adapter_path)
    model = model.to(device).eval()

    name = args.dataset
    summary = {"dataset": args.dataset, "config_name": args.config_name, "seed": args.seed, "splits": {}}
    for split in FOBLIMP_SPLITS if name == "FoBLiMP" else FOBCOMP_SPLITS:
        ds = load_split(args.dataset, split)
        rows = [{"index": i, **score_pair(model, tokenizer, ex["correct_text"], ex["wrong_text"], device)}
                for i, ex in enumerate(tqdm(ds, desc=f"{name}/{split}"))]
        n_correct = sum(r["correct"] for r in rows)
        summary["splits"][split] = {"correct": n_correct, "total": len(rows), "accuracy": n_correct / len(rows)}
        print(f"{split}: {n_correct}/{len(rows)} = {n_correct / len(rows):.4f}")
        with open(os.path.join(args.output_dir, f"{name}_{split}_{args.config_name}_seed{args.seed}.jsonl"), "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    with open(os.path.join(args.output_dir, f"{name}_{args.config_name}_seed{args.seed}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
