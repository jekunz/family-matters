#!/bin/bash
#SBATCH -t 3-00:00:00
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%j.out
#
# One continued pre-training run (either stage). Optional: INIT (model or
# adapter to continue from), SEED (default 1), LR (default 5e-4 full / 8e-4 LoRA).
#   SIZE=360m METHOD=full DATA=is EPOCHS=1 OUT=smollm-360m-full-is bash scripts/train.sh
source scripts/common.sh
: "${SIZE:?}" "${METHOD:?}" "${DATA:?}" "${EPOCHS:?}" "${OUT:?}"
case "${METHOD}" in full) DEFAULT_LR=5e-4 ;; lora) DEFAULT_LR=8e-4 ;; *) echo "Unknown METHOD=${METHOD}" >&2; exit 1 ;; esac
OUTPUT_DIR="${MODELS_DIR}/${OUT}"
mkdir -p "${OUTPUT_DIR}"

echo "Training ${OUT}: size=${SIZE} method=${METHOD} data=${DATA} epochs=${EPOCHS} init=${INIT:-<base>} seed=${SEED:-1}"
"${LAUNCH[@]}" train/train.py --base_model "$(base_model_for "${SIZE}")" --method "${METHOD}" \
    ${INIT:+--init "${INIT}"} --dataset_path "${DATA_DIR}/${DATA}" --num_epochs "${EPOCHS}" \
    --learning_rate "${LR:-${DEFAULT_LR}}" --seed "${SEED:-1}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --output_dir "${OUTPUT_DIR}"
echo "Done: ${OUTPUT_DIR}"
