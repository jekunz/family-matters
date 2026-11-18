"""Tokenization of FineWeb-2 into 2048-token chunks."""

import logging

from datasets import DatasetDict, load_dataset

logger = logging.getLogger(__name__)

MAX_SEQ_LENGTH = 2048
VALIDATION_SPLIT = 0.01
SPLIT_SEED = 42


def tokenize_fineweb(tokenizer, dataset_config, max_samples=None, num_workers=8):
    """Tokenize one FineWeb-2 language (1% held out) into MAX_SEQ_LENGTH-token chunks."""
    dataset = load_dataset("HuggingFaceFW/fineweb-2", name=dataset_config, split="train")
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info(f"Using the first {len(dataset):,} documents")
    split = dataset.train_test_split(test_size=VALIDATION_SPLIT, seed=SPLIT_SEED)
    dataset = DatasetDict({"train": split["train"], "validation": split["test"]})

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=False, padding=False)

    def group(examples):
        concatenated = {k: sum(examples[k], []) for k in examples}
        total = len(concatenated["input_ids"])
        if total >= MAX_SEQ_LENGTH:
            total = (total // MAX_SEQ_LENGTH) * MAX_SEQ_LENGTH
        result = {k: [t[i:i + MAX_SEQ_LENGTH] for i in range(0, total, MAX_SEQ_LENGTH)]
                  for k, t in concatenated.items()}
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = dataset.map(tokenize, batched=True, num_proc=num_workers,
                          remove_columns=dataset["train"].column_names, desc="Tokenizing")
    return dataset.map(group, batched=True, num_proc=num_workers, desc="Grouping into chunks")

