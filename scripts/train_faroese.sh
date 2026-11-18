#!/bin/bash
#SBATCH -t 3-00:00:00
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%j.out
#
# Faroese stage 2 (5 epochs) from a stage-1 initialization on the Hub (HUB_NAMESPACE).
# SOURCE_TYPE/SOURCE_NAME: baseline/none, individual/<language>, merge/{equal,isplus,da_is},
# multilingual/joint (train it first, see README).
#   SIZE=360m METHOD=lora SOURCE_TYPE=individual SOURCE_NAME=icelandic SEED=1 bash scripts/train_faroese.sh
source scripts/common.sh
: "${SIZE:?}" "${METHOD:?}" "${SOURCE_TYPE:?}" "${SOURCE_NAME:?}" "${SEED:?}"
NS="${HUB_NAMESPACE:-jekunz}"
N="${SIZE%m}"

case "${SOURCE_TYPE}/${METHOD}" in
  baseline/*)          INIT="" ;;
  individual/full)     INIT="${NS}/smollm-${SIZE}-cpt-fineweb-${SOURCE_NAME}" ;;
  individual/lora)     INIT="${NS}/smollm-${SIZE}-lora-fineweb-${SOURCE_NAME}" ;;
  merge/full)
    case "${SOURCE_NAME}" in
      equal)  INIT="${NS}/smollm${SIZE}-da1-is1-nno1-nob1-sv1-ties" ;;
      isplus) INIT="${NS}/smollm${SIZE}-da05-is1-nno05-nob05-sv05-ties" ;;
      da_is)  INIT="${NS}/smollm${SIZE}-da1-is1-ties" ;;
    esac ;;
  merge/lora)
    case "${SOURCE_NAME}" in
      equal)  INIT="${NS}/new_scandimerge_equal_${N}" ;;
      isplus) INIT="${NS}/new_scandimerge_isplus_${N}" ;;
      da_is)  INIT="${NS}/new_scandimerge_2_${N}" ;;
    esac ;;
  multilingual/*)      INIT="${MODELS_DIR}/smollm-${SIZE}-${METHOD}-multilingual-baseline-seed${MULTILINGUAL_SEED:-1}" ;;
  *) echo "Unknown SOURCE_TYPE=${SOURCE_TYPE}" >&2; exit 1 ;;
esac
[[ "${SOURCE_TYPE}" == baseline || -n "${INIT:-}" ]] || { echo "Unknown SOURCE_NAME=${SOURCE_NAME}" >&2; exit 1; }

export SIZE METHOD SEED INIT DATA=fao EPOCHS=5 OUT="smollm-${SIZE}-${METHOD}-fao-${SOURCE_TYPE}-${SOURCE_NAME}-seed${SEED}"
exec bash scripts/train.sh
