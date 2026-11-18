"""
TIES-merge LoRA adapters (density 0.5) with PEFT.

Usage (Merge^is+, Icelandic first; Merge^eq: all weights 1; Merge^da+is: Icelandic, Danish, weights 1 1):
    python merge/merge_lora_adapters.py --base_model HuggingFaceTB/SmolLM2-360M-Instruct \\
        --adapters jekunz/smollm-360m-lora-fineweb-icelandic jekunz/smollm-360m-lora-fineweb-danish \\
                   jekunz/smollm-360m-lora-fineweb-norwegian-bokmaal jekunz/smollm-360m-lora-fineweb-norwegian-nynorsk \\
                   jekunz/smollm-360m-lora-fineweb-swedish \\
        --weights 1 0.5 0.5 0.5 0.5 --output_dir models/lora-merge-isplus-360m
"""

import argparse

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapters", nargs="+", required=True, help="Adapter paths or Hub ids")
    p.add_argument("--weights", nargs="+", type=float, default=None, help="One weight per adapter (default: all 1)")
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    weights = args.weights or [1.0] * len(args.adapters)
    if len(weights) != len(args.adapters):
        p.error("--weights needs one value per adapter")

    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    names = [f"a{i}" for i in range(len(args.adapters))]
    model = PeftModel.from_pretrained(model, args.adapters[0], adapter_name=names[0])
    for name, path in zip(names[1:], args.adapters[1:]):
        model.load_adapter(path, adapter_name=name)

    model.add_weighted_adapter(names, weights, "merged", combination_type="ties", density=0.5)
    model.set_adapter("merged")
    for name in names:
        model.delete_adapter(name)
    model.save_pretrained(args.output_dir)   # PEFT writes a non-default adapter to <output_dir>/merged
    AutoTokenizer.from_pretrained(args.base_model).save_pretrained(f"{args.output_dir}/merged")
    print(f"Saved merged adapter (weights {weights}) to {args.output_dir}/merged")


if __name__ == "__main__":
    main()
