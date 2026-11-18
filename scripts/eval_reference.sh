#!/bin/bash
#SBATCH -t 0-08:00:00
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%j.out
#
# Apertus-8B reference: minimal pairs and MT.
source scripts/common.sh
python evaluate/evaluate_reference.py --output_dir "${RESULTS_DIR}/reference_llms"
