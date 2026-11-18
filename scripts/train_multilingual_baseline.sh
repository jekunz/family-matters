#!/bin/bash
#SBATCH -t 3-00:00:00
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%j.out
#
# Stage 1 of the joint multilingual baseline (one epoch).
#   SIZE=360m METHOD=full bash scripts/train_multilingual_baseline.sh
source scripts/common.sh
: "${SIZE:?}" "${METHOD:?}"
export SIZE METHOD SEED="${SEED:-1}" INIT="" DATA=multilingual_baseline EPOCHS=1 \
       OUT="smollm-${SIZE}-${METHOD}-multilingual-baseline-seed${SEED:-1}"
exec bash scripts/train.sh
