# Sourced by the other scripts; run them from the repository root.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f train/train.py ]] || { echo "Run this from the repository root" >&2; exit 1; }
source scripts/env.sh
set -euo pipefail
mkdir -p logs

base_model_for() {
  case "$1" in
    135m) echo "HuggingFaceTB/SmolLM2-135M-Instruct" ;;
    360m) echo "HuggingFaceTB/SmolLM2-360M-Instruct" ;;
    *) echo "Unknown SIZE=$1 (expected 135m or 360m)" >&2; exit 1 ;;
  esac
}

# python, or torchrun when NUM_GPUS > 1
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  LAUNCH=(torchrun --nproc_per_node="${NUM_GPUS}" --master_port=$((20000 + ${SLURM_JOB_ID:-$$} % 20000)))
else
  LAUNCH=(python)
fi

# Effective batch size 256 for any number of GPUs.
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
GRAD_ACCUM=$((256 / (NUM_GPUS * PER_DEVICE_BATCH_SIZE)))
