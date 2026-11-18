#!/bin/bash
#SBATCH -t 0-12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/%x_%j.out
#
# Pre-tokenize one FineWeb-2 language (or "multilingual") into ${DATA_DIR}/<lang>.
#   bash scripts/preprocess.sh fao
source scripts/common.sh
LANG_CODE="${1:-${LANG_CODE:?usage: preprocess.sh <lang>}}"

if [[ "${LANG_CODE}" == "multilingual" ]]; then
  python preprocess/preprocess_multilingual_baseline.py --output_dir "${DATA_DIR}/multilingual_baseline"
  exit 0
fi

case "${LANG_CODE}" in
  fao) CONFIG=fao_Latn ;; da) CONFIG=dan_Latn ;; is) CONFIG=isl_Latn ;; nb) CONFIG=nob_Latn ;;
  nn) CONFIG=nno_Latn ;; sv) CONFIG=swe_Latn ;; sme) CONFIG=sme_Latn ;; fi) CONFIG=fin_Latn ;;
  et) CONFIG=ekk_Latn ;; hsb) CONFIG=hsb_Latn ;; cs) CONFIG=ces_Latn ;; de) CONFIG=deu_Latn ;; pl) CONFIG=pol_Latn ;;
  *) echo "Unknown language ${LANG_CODE}" >&2; exit 1 ;;
esac
# 10M documents are enough for the 4B-token cap.
case "${LANG_CODE}" in
  da|nb|sv|fi|et|cs|de|pl) MAX_SAMPLES=(--max_samples 10000000) ;;
  *) MAX_SAMPLES=() ;;
esac

python preprocess/preprocess_data.py --dataset_config "${CONFIG}" --output_dir "${DATA_DIR}/${LANG_CODE}" \
    ${MAX_SAMPLES[@]+"${MAX_SAMPLES[@]}"}
