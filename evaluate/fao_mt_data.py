"""
English->Faroese MT data (FLORES+, BOUQuET; both gated, need HF_TOKEN),
few-shot prompt and chrF++ scoring.
"""

import random

import sacrebleu
from datasets import load_dataset

K_SHOT = 5
FEWSHOT_SEED = 0


def load_flores_pairs(split):
    """[(id, english, faroese)] for one FLORES+ split ('dev' or 'devtest')."""
    en = load_dataset("openlanguagedata/flores_plus", "eng_Latn", split=split)
    fao = {row["id"]: row["text"] for row in load_dataset("openlanguagedata/flores_plus", "fao_Latn", split=split)}
    return [(row["id"], row["text"], fao[row["id"]]) for row in en if row["id"] in fao]


def load_bouquet_pairs(split):
    """[(index, english, faroese)] for one BOUQuET split."""
    ds = load_dataset("facebook/bouquet", "sentence_level", split=split)
    rows = ds.filter(lambda ex: ex["src_lang"] == "fao_Latn")
    return [(i, row["tgt_text"], row["src_text"]) for i, row in enumerate(rows)]


def fewshot_examples():
    return random.Random(FEWSHOT_SEED).sample(load_flores_pairs("dev"), K_SHOT)


def test_sets():
    return {"flores_devtest": load_flores_pairs("devtest"), "bouquet_test": load_bouquet_pairs("test")}


def fewshot_prompt(fewshot, src_text):
    examples = "\n\n".join(f"English: {en}\nFaroese: {fo}" for _, en, fo in fewshot)
    return f"{examples}\n\nEnglish: {src_text}\nFaroese:"


def first_line(text):
    """Cut a generation at the next 'English:' turn, then keep its first line."""
    return text.split("English:")[0].strip().split("\n")[0].strip()


def chrf_pp(hypotheses, references):
    """Corpus-level chrF++ (chrF with word bigrams)."""
    return sacrebleu.corpus_chrf(hypotheses, [references], word_order=2).score
