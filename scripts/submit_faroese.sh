#!/bin/bash
# Submit all 120 Faroese runs (MODE=train or MODE=eval). SBATCH_ARGS goes to sbatch; DRY_RUN=1 prints.
set -euo pipefail
MODE="${MODE:?set MODE=train or MODE=eval}"
case "${MODE}" in
  train) SCRIPT=scripts/train_faroese.sh ;;
  eval)  SCRIPT=scripts/eval_faroese.sh ;;
  *) echo "MODE must be train or eval" >&2; exit 1 ;;
esac
SOURCES=("baseline none" "individual danish" "individual icelandic" "individual norwegian-bokmaal"
         "individual norwegian-nynorsk" "individual swedish" "merge equal" "merge isplus" "merge da_is"
         "multilingual joint")
submit() { if [[ -n "${DRY_RUN:-}" ]]; then echo sbatch ${SBATCH_ARGS:-} --export="$1" "${SCRIPT}";
           else sbatch ${SBATCH_ARGS:-} --export="$1" "${SCRIPT}"; fi; }
for SIZE in 135m 360m; do
  [[ "${MODE}" == eval ]] && submit "ALL,SIZE=${SIZE},SOURCE_TYPE=untuned"
  for METHOD in full lora; do
    for SRC in "${SOURCES[@]}"; do
      read -r ST SN <<< "${SRC}"
      for SEED in 1 2 3; do
        submit "ALL,SIZE=${SIZE},METHOD=${METHOD},SOURCE_TYPE=${ST},SOURCE_NAME=${SN},SEED=${SEED}"
      done
    done
  done
done
