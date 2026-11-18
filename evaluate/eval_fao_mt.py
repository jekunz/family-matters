"""
Five-shot English->Faroese MT (FLORES+ devtest, BOUQuET test), greedy
decoding, corpus-level chrF++.

Usage:
    python evaluate/eval_fao_mt.py --model_name_or_path path/to/checkpoint \\
        --config_name individual-icelandic-360m-full --seed 1 --output_dir results/fao_mt
"""

import argparse
import json
import os

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from fao_mt_data import chrf_pp, fewshot_examples, fewshot_prompt, first_line, test_sets

MAX_NEW_TOKENS = 256
BATCH_SIZE = 16


def translate(model, tokenizer, device, prompts):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                       max_length=2048 - MAX_NEW_TOKENS).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    # With left padding, every continuation starts at the same index.
    texts = tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return [first_line(t) for t in texts]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name_or_path", default=None, help="Full model checkpoint")
    p.add_argument("--base_model", default=None, help="Base model, when loading a LoRA adapter")
    p.add_argument("--adapter_path", default=None, help="LoRA adapter checkpoint")
    p.add_argument("--config_name", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()
    if not args.model_name_or_path and not (args.base_model and args.adapter_path):
        p.error("pass either --model_name_or_path, or both --base_model and --adapter_path")
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fewshot, tests = fewshot_examples(), test_sets()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path or args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    else:
        model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(args.base_model), args.adapter_path)
    model = model.to(device).eval()

    summary = []
    for name, pairs in tests.items():
        rows = []
        for start in tqdm(range(0, len(pairs), BATCH_SIZE), desc=name):
            batch = pairs[start:start + BATCH_SIZE]
            hyps = translate(model, tokenizer, device, [fewshot_prompt(fewshot, src) for _, src, _ in batch])
            rows += [{"id": idx, "src": src, "ref": ref, "hyp": hyp} for (idx, src, ref), hyp in zip(batch, hyps)]
        with open(os.path.join(args.output_dir, f"{name}_{args.config_name}_seed{args.seed}_translations.jsonl"), "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        score = chrf_pp([r["hyp"] for r in rows], [r["ref"] for r in rows])
        summary.append({"config_name": args.config_name, "seed": args.seed, "dataset": name,
                        "n_sentences": len(rows), "chrf2": score})
        print(f"{name}: chrF++ {score:.2f}")

    with open(os.path.join(args.output_dir, f"mt_summary_{args.config_name}_seed{args.seed}.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
