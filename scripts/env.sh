# Edit once for your machine; every variable can be overridden from the shell.

export DATA_DIR="${DATA_DIR:-${PWD}/data}"          # pre-tokenized corpora
export MODELS_DIR="${MODELS_DIR:-${PWD}/models}"    # trained checkpoints
export RESULTS_DIR="${RESULTS_DIR:-${PWD}/results}" # evaluation outputs
export NUM_GPUS="${NUM_GPUS:-1}"                    # GPUs per training job
# export HF_HOME=/path/to/hf_cache                  # optional: shared Hugging Face cache
# export HF_TOKEN=...                               # needed for the gated FLORES+/BOUQuET datasets

# Activate your Python environment here, for example:
# source /path/to/venv/bin/activate
# conda activate family-matters
