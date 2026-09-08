#!/bin/bash

set -euo pipefail

MODE=submit
if [ "${1:-}" = "--dry-run" ]
then
  MODE=dry-run
elif [ "$#" -ne 0 ]
then
  echo "Usage: $0 [--dry-run]"
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
RELNET_SCRATCH="${RELNET_SCRATCH:-/home/ucabggs/Scratch}"
PREFERENCE_SOURCE="${PREFERENCE_SOURCE:-${SCRIPT_DIR}}"
PREFERENCE_DATA="${PREFERENCE_DATA:-${RELNET_SCRATCH}/experiment_data/relnet_preference_s2v}"
LOG_DIR="${LOG_DIR:-${RELNET_SCRATCH}/relnet_preference_s2v_logs}"
RECORD="${PREFERENCE_SOURCE}/conditioned_extra_seed_submissions.tsv"
JOB_SCRIPT="${PREFERENCE_SOURCE}/relnet_cpu_preference_conditioned.sh"
EXPERIMENT_VARIANT=s2v_full400k
AGENT_BUDGET=400000
TRAINING_WEIGHTS=0.0:0.1:0.2:0.3:0.4:0.5:0.6:0.7:0.8:0.9:1.0
VALIDATION_INTERVAL=5000
SEEDS=(42 84 126 168)
SETTINGS=(
  "10 2.5 2"
  "10 5 3"
  "10 10 5"
  "20 1 2"
  "20 2.5 5"
  "20 5 10"
)
GENERATORS=(random_network barabasi_albert)

if [ "$MODE" = submit ]
then
  if [ ! -f "$JOB_SCRIPT" ]
  then
    echo "Missing job script: $JOB_SCRIPT"
    exit 2
  fi
  mkdir -p "$LOG_DIR"
  if [ ! -e "$RECORD" ]
  then
    printf 'submitted_at\tjob_id\tn\tL\tgenerator\tmodel_seed\texperiment_id\n' > "$RECORD"
  fi
fi

seed_tag() {
  case "$1" in
    42) echo 1 ;;
    84) echo 2 ;;
    126) echo 3 ;;
    168) echo 4 ;;
    *) return 1 ;;
  esac
}

generator_tag() {
  case "$1" in
    random_network) echo ER ;;
    barabasi_albert) echo BA ;;
    *) return 1 ;;
  esac
}

submitted=0
skipped_complete=0
skipped_recorded=0
considered=0

for setting in "${SETTINGS[@]}"
do
  read -r graph_n edge_percentage edge_budget <<< "$setting"
  experiment_id="relnet_cpu_pref_${EXPERIMENT_VARIANT}_n${graph_n}_L${edge_budget}_s1exact"

  for generator in "${GENERATORS[@]}"
  do
    generator_short="$(generator_tag "$generator")"

    for model_seed in "${SEEDS[@]}"
    do
      considered=$((considered + 1))
      model_path="${PREFERENCE_DATA}/${experiment_id}/models/checkpoints/rnet_dqn_pref-combined_linear-${generator}-${model_seed}-0/rnet_dqn_pref_agent.model"
      manifest_path="${PREFERENCE_DATA}/${experiment_id}/complete_preference_${generator}_seed${model_seed}.json"
      key_pattern=$(printf '\t%s\t%s\t%s\t%s\t%s$' "$graph_n" "$edge_budget" "$generator" "$model_seed" "$experiment_id")

      if [ -s "$model_path" ] && [ -s "$manifest_path" ]
      then
        echo "SKIPPING completed: n=$graph_n L=$edge_budget generator=$generator seed=$model_seed"
        skipped_complete=$((skipped_complete + 1))
        continue
      fi

      if [ "$MODE" = submit ] && grep -Fq "$key_pattern" "$RECORD"
      then
        echo "SKIPPING already recorded: n=$graph_n L=$edge_budget generator=$generator seed=$model_seed"
        skipped_recorded=$((skipped_recorded + 1))
        continue
      fi

      seed_short="$(seed_tag "$model_seed")"
      job_name="c${graph_n}L${edge_budget}${generator_short}s${seed_short}"
      variables="WHICH=synth,GRAPH_N=${graph_n},EDGE_PERCENTAGE=${edge_percentage},EDGE_BUDGET=${edge_budget},NETWORK_GENERATOR=${generator},MODEL_SEED=${model_seed},AGENT_BUDGET=${AGENT_BUDGET},EXPERIMENT_VARIANT=${EXPERIMENT_VARIANT},TRAINING_WEIGHTS=${TRAINING_WEIGHTS},VALIDATION_INTERVAL=${VALIDATION_INTERVAL}"

      if [ "$MODE" = dry-run ]
      then
        echo "WOULD SUBMIT: $job_name n=$graph_n L=$edge_budget generator=$generator seed=$model_seed"
      else
        job_id=$(qsub -terse -N "$job_name" -o "$LOG_DIR" -j y -v "$variables" "$JOB_SCRIPT")
        submitted_at=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "$submitted_at" "$job_id" "$graph_n" "$edge_budget" \
          "$generator" "$model_seed" "$experiment_id" >> "$RECORD"
        echo "Submitted $job_name: $job_id"
      fi
      submitted=$((submitted + 1))
    done
  done
done

if [ "$considered" -ne 48 ]
then
  echo "Internal error: expected 48 jobs, considered $considered"
  exit 1
fi

if [ "$MODE" = dry-run ]
then
  echo "Dry run complete: $submitted jobs would be submitted."
else
  echo "Sweep complete: submitted=$submitted completed=$skipped_complete already_recorded=$skipped_recorded total=$considered"
  echo "Submission record: $RECORD"
fi
