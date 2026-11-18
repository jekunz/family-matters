"""
Apertus-8B-Instruct-2509 on FoBLiMP/FoBCoMP and five-shot MT (needs
transformers>=4.56). As in the paper, at most 500 examples per minimal-pair
split and 200 MT sentences; pass 0 for the full sets.

Usage:
    python evaluate/evaluate_reference.py --output_dir results/reference_llms
"""

import argparse
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval_minimal_pairs import FOBCOMP_SPLITS, FOBLIMP_SPLITS, load_split, score_pair
from fao_mt_data import chrf_pp, fewshot_examples, fewshot_prompt, test_sets

MODEL_KEY = "apertus-8b-instruct-2509"
REPO_ID = "swiss-ai/Apertus-8B-Instruct-2509"
MT_INSTRUCTION = ("Translate the following English sentences into Faroese. "
                  "Output only the Faroese translation and nothing else.\n\n")
MAX_NEW_TOKENS = 256
BATCH_SIZE = 8


def minimal_pairs(model, tokenizer, device, max_examples, output_dir):
    summary = []
    jobs = [("FoBLiMP", s) for s in FOBLIMP_SPLITS] + [("FoBCoMP", s) for s in FOBCOMP_SPLITS]
    for dataset, split in jobs:
        ds = load_split(dataset, split)
        if max_examples:
            ds = ds.select(range(min(max_examples, len(ds))))
        rows = [{"index": i, **score_pair(model, tokenizer, ex["correct_text"], ex["wrong_text"], device)}
                for i, ex in enumerate(tqdm(ds, desc=f"{dataset}/{split}"))]
        with open(os.path.join(output_dir, f"{MODEL_KEY}_{dataset}_{split}.jsonl"), "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        accuracy = sum(r["correct"] for r in rows) / len(rows)
        summary.append({"model": MODEL_KEY, "dataset": dataset, "split": split,
                        "n_items": len(rows), "mean_accuracy": accuracy})
        print(f"{dataset}/{split}: {accuracy:.4f} (n={len(rows)})")
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def translate(model, tokenizer, device, prompts):
    formatted = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                               add_generation_prompt=True) for p in prompts]
    inputs = tokenizer(formatted, return_tensors="pt", padding=True, truncation=True,
                       max_length=2048 - MAX_NEW_TOKENS, add_special_tokens=False).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    texts = tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return [t.strip().split("\n")[0].strip() for t in texts]


def machine_translation(model, tokenizer, device, max_examples, output_dir):
    fewshot, summary = fewshot_examples(), []
    for name, pairs in test_sets().items():
        pairs = pairs[:max_examples] if max_examples else pairs
        rows = []
        for start in tqdm(range(0, len(pairs), BATCH_SIZE), desc=name):
            batch = pairs[start:start + BATCH_SIZE]
            hyps = translate(model, tokenizer, device,
                             [MT_INSTRUCTION + fewshot_prompt(fewshot, src) for _, src, _ in batch])
            rows += [{"id": idx, "src": src, "ref": ref, "hyp": hyp} for (idx, src, ref), hyp in zip(batch, hyps)]
        with open(os.path.join(output_dir, f"{MODEL_KEY}_{name}_translations.jsonl"), "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        score = chrf_pp([r["hyp"] for r in rows], [r["ref"] for r in rows])
        summary.append({"model": MODEL_KEY, "dataset": name, "n_sentences": len(rows), "chrf2": score})
        print(f"{name}: chrF++ {score:.2f} (n={len(rows)})")
    with open(os.path.join(output_dir, "mt_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max_examples_per_split", type=int, default=500, help="Minimal pairs; 0 = all")
    p.add_argument("--max_mt_sentences", type=int, default=200, help="0 = all")
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        REPO_ID, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()

    minimal_pairs(model, tokenizer, device, args.max_examples_per_split, args.output_dir)
    machine_translation(model, tokenizer, device, args.max_mt_sentences, args.output_dir)


if __name__ == "__main__":
    main()
