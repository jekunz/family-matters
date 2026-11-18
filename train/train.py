"""
Continued pre-training of SmolLM2-Instruct: full fine-tuning (--method full),
a new LoRA adapter (--method lora), or continuing an adapter (--method lora --init).

Usage:
    python train/train.py --base_model HuggingFaceTB/SmolLM2-360M-Instruct --method lora \\
        --init jekunz/smollm-360m-lora-fineweb-icelandic --dataset_path data/fao \\
        --num_epochs 5 --learning_rate 8e-4 --seed 1 --output_dir models/example
"""

import argparse
import logging
import sys

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, default_data_collator

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler(sys.stdout)],
                    level=logging.INFO)
logger = logging.getLogger(__name__)

LORA_TARGET_MODULES = ["up_proj", "down_proj", "gate_proj", "o_proj"]


def load_dataset(path, max_tokens):
    """Load a preprocessed dataset, keeping the first chunks up to ~max_tokens."""
    dataset = load_from_disk(path, keep_in_memory=False)
    sample = dataset["train"].select(range(min(100, len(dataset["train"]))))
    avg_tokens = sum(len(ids) for ids in sample["input_ids"]) / len(sample)
    n = min(int(max_tokens / avg_tokens), len(dataset["train"]))
    logger.info(f"Using {n:,} of {len(dataset['train']):,} chunks (~{n * avg_tokens:,.0f} tokens)")
    dataset["train"] = dataset["train"].select(range(n), keep_in_memory=False)
    return dataset


def load_model(args):
    if args.method == "full":
        return AutoModelForCausalLM.from_pretrained(args.init or args.base_model, torch_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    if args.init:
        return PeftModel.from_pretrained(base, args.init, is_trainable=True)
    config = LoraConfig(r=256, lora_alpha=512, lora_dropout=0.05, target_modules=LORA_TARGET_MODULES,
                        task_type=TaskType.CAUSAL_LM, bias="none")
    return get_peft_model(base, config)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_model", required=True, help="SmolLM2-{135M,360M}-Instruct")
    p.add_argument("--method", required=True, choices=["full", "lora"])
    p.add_argument("--init", default=None, help="Full model or LoRA adapter to continue from (path or Hub id)")
    p.add_argument("--dataset_path", required=True, help="Output of preprocess_data.py")
    p.add_argument("--max_tokens", type=int, default=4_000_000_000)
    p.add_argument("--num_epochs", type=int, required=True)
    p.add_argument("--learning_rate", type=float, required=True)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=32)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(args)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable:,} / {sum(p.numel() for p in model.parameters()):,}")
    dataset = load_dataset(args.dataset_path, args.max_tokens)

    # Continuing an existing adapter used no weight decay and no gradient checkpointing.
    continue_adapter = args.method == "lora" and args.init is not None
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.0 if continue_adapter else 0.01,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=not continue_adapter,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_steps=1000,
        save_total_limit=3,
        seed=args.seed,
        report_to="none",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=dataset["train"],
                      processing_class=tokenizer, data_collator=default_data_collator)
    result = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
