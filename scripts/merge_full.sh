#!/bin/bash
#SBATCH -t 0-02:00:00
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%j.out
#
# TIES-merge full models with mergekit.
#   bash scripts/merge_full.sh merge/mergekit_configs/fao_merge_isplus_360.yaml smollm-360m-merge-isplus
source scripts/common.sh
CONFIG="${1:?usage: merge_full.sh <config.yaml> <output name>}"
OUT="${2:?usage: merge_full.sh <config.yaml> <output name>}"
TMP="$(mktemp "${TMPDIR:-/tmp}/merge-config.XXXXXX")"
envsubst '${MODELS_DIR}' < "${CONFIG}" > "${TMP}"
mergekit-yaml "${TMP}" "${MODELS_DIR}/${OUT}" --copy-tokenizer
rm -f "${TMP}"
