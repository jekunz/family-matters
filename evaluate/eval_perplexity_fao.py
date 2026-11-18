"""
Token-weighted perplexity on the Faroese FineWeb-2 test split (documents
truncated at 2048 tokens).

Usage:
    python evaluate/eval_perplexity_fao.py --model_name_or_path path/to/checkpoint \\
        --config_name individual-icelandic-360m-full --seed 1 --output_dir results/perplexity
"""

import argparse
import json
import math
import os

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name_or_path", default=None)
    p.add_argument("--base_model", default=None)
    p.add_argument("--adapter_path", default=None)
    p.add_argument("--config_name", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()
    if not args.model_name_or_path and not (args.base_model and args.adapter_path):
        p.error("pass either --model_name_or_path, or both --base_model and --adapter_path")
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path or args.base_model)
    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    else:
        model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(args.base_model), args.adapter_path)
    model = model.to(device).eval()

    texts = [ex["text"] for ex in load_dataset("HuggingFaceFW/fineweb-2", name="fao_Latn", split="test", streaming=True)]
    rows = []
    for i, text in enumerate(tqdm(texts, desc="perplexity")):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
        n = inputs["input_ids"].size(1)
        if n < 2:
            continue
        with torch.no_grad():
            loss = model(**inputs, labels=inputs["input_ids"]).loss.item()
        # loss is the mean NLL over the n - 1 predicted tokens
        rows.append({"index": i, "loss": loss, "num_tokens": n - 1})

    total_tokens = sum(r["num_tokens"] for r in rows)
    perplexity = math.exp(sum(r["loss"] * r["num_tokens"] for r in rows) / total_tokens)
    print(f"Token-weighted perplexity over {len(rows)} documents / {total_tokens} tokens: {perplexity:.4f}")

    prefix = os.path.join(args.output_dir, f"perplexity_fao_Latn_{args.config_name}_seed{args.seed}")
    with open(prefix + ".jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    with open(prefix + "_summary.json", "w") as f:
        json.dump({"config_name": args.config_name, "seed": args.seed, "num_documents": len(rows),
                   "total_tokens": total_tokens, "perplexity": perplexity}, f, indent=2)


if __name__ == "__main__":
    main()
