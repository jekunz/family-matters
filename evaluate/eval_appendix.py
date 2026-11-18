"""
Northern Sámi (sme) / Upper Sorbian (hsb) evaluation as in the appendix:
perplexity is exp of the mean per-document loss; MultiBLiMP uses summed
log-probabilities, MultiWikiQA per-token perplexity.

Usage:
    python evaluate/eval_appendix.py --lang hsb --model_name_or_path path/to/checkpoint --output results/appendix/x.json
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

DATA = {
    "hsb": {"fineweb": "hsb_Latn", "multiblimp": "hsb", "wikiqa": "liu-nlp/multi_wiki_qa_hsb_minpairs_shuffled"},
    "sme": {"fineweb": "sme_Latn", "multiblimp": "sme", "wikiqa": "liu-nlp/multi_wiki_qa_se_minpairs_shuffled"},
}


def loss_and_len(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(model.device)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss.item()
    return loss, inputs["input_ids"].size(1)


def perplexity(model, tokenizer, config):
    ds = load_dataset("HuggingFaceFW/fineweb-2", name=config, split="test", streaming=True)
    texts = [ex["text"] for ex in ds]
    losses = [loss_and_len(model, tokenizer, t)[0] for t in tqdm(texts, desc="perplexity")]
    return math.exp(sum(losses) / len(losses)), len(texts)


def multiblimp(model, tokenizer, lang):
    ds = load_dataset("jumelet/multiblimp", lang, split="train")
    correct = 0
    for ex in tqdm(ds, desc="MultiBLiMP"):
        lg, ng = loss_and_len(model, tokenizer, ex["sen"])
        lb, nb = loss_and_len(model, tokenizer, ex["wrong_sen"])
        correct += -lg * ng > -lb * nb
    return correct / len(ds), len(ds)


def wikiqa(model, tokenizer, repo):
    ds = load_dataset(repo, split="train")
    correct = 0
    for ex in tqdm(ds, desc="MultiWikiQA"):
        correct += loss_and_len(model, tokenizer, ex["correct_text"])[0] < loss_and_len(model, tokenizer, ex["wrong_text"])[0]
    return correct / len(ds), len(ds)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lang", required=True, choices=sorted(DATA))
    p.add_argument("--model_name_or_path", default=None)
    p.add_argument("--base_model", default=None)
    p.add_argument("--adapter_path", default=None)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if not args.model_name_or_path and not (args.base_model and args.adapter_path):
        p.error("pass either --model_name_or_path, or both --base_model and --adapter_path")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.base_model or args.model_name_or_path)
    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    else:
        model = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(args.base_model), args.adapter_path)
    model = model.to(device).eval()

    d, out = DATA[args.lang], vars(args)
    out["perplexity"], out["perplexity_n"] = perplexity(model, tokenizer, d["fineweb"])
    out["multiblimp_accuracy"], out["multiblimp_n"] = multiblimp(model, tokenizer, d["multiblimp"])
    out["multiwikiqa_accuracy"], out["multiwikiqa_n"] = wikiqa(model, tokenizer, d["wikiqa"])
    print(json.dumps(out, indent=2))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
