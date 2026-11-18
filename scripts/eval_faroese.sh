#!/bin/bash
#SBATCH -t 0-12:00:00
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%j.out
#
# Evaluate one checkpoint from train_faroese.sh (SOURCE_TYPE=untuned: SmolLM2-Instruct).
#   SIZE=360m METHOD=lora SOURCE_TYPE=individual SOURCE_NAME=icelandic SEED=1 bash scripts/eval_faroese.sh
source scripts/common.sh
: "${SIZE:?}" "${SOURCE_TYPE:?}"
BASE_MODEL="$(base_model_for "${SIZE}")"
TASKS="${TASKS:-minimal_pairs,perplexity,mt}"

if [[ "${SOURCE_TYPE}" == "untuned" ]]; then
  CONFIG="untuned-smollm2-${SIZE}"; SEED=0
  MODEL_ARGS=(--model_name_or_path "${BASE_MODEL}")
else
  : "${METHOD:?}" "${SOURCE_NAME:?}" "${SEED:?}"
  CONFIG="${SOURCE_TYPE}-${SOURCE_NAME}-${SIZE}-${METHOD}"
  CKPT="${MODELS_DIR}/smollm-${SIZE}-${METHOD}-fao-${SOURCE_TYPE}-${SOURCE_NAME}-seed${SEED}"
  if [[ "${METHOD}" == "full" ]]; then
    MODEL_ARGS=(--model_name_or_path "${CKPT}")
  else
    MODEL_ARGS=(--base_model "${BASE_MODEL}" --adapter_path "${CKPT}")
  fi
fi
echo "Evaluating ${CONFIG} seed ${SEED}: ${TASKS}"

if [[ ",${TASKS}," == *",minimal_pairs,"* ]]; then
  for DS in FoBLiMP:foblimp FoBCoMP:fobcomp; do
    python evaluate/eval_minimal_pairs.py --dataset "${DS%%:*}" "${MODEL_ARGS[@]}" --tokenizer_name "${BASE_MODEL}" \
        --config_name "${CONFIG}" --seed "${SEED}" --output_dir "${RESULTS_DIR}/${DS##*:}"
  done
fi
if [[ ",${TASKS}," == *",perplexity,"* && "${SOURCE_TYPE}" != "untuned" ]]; then
  python evaluate/eval_perplexity_fao.py "${MODEL_ARGS[@]}" --config_name "${CONFIG}" --seed "${SEED}" \
      --output_dir "${RESULTS_DIR}/perplexity"
fi
if [[ ",${TASKS}," == *",mt,"* && "${SOURCE_TYPE}" != "untuned" ]]; then
  python evaluate/eval_fao_mt.py "${MODEL_ARGS[@]}" --config_name "${CONFIG}" --seed "${SEED}" \
      --output_dir "${RESULTS_DIR}/fao_mt"
fi
