#!/bin/bash

set -euo pipefail

GRAPH_N=20
EDGE_PERCENTAGE=2.5
EDGE_BUDGET=5
AGENT_BUDGET="${AGENT_BUDGET:-400000}"
VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-5000}"
EXPERIMENT_VARIANT="${EXPERIMENT_VARIANT:-density_v1}"
TRAINING_WEIGHTS="${TRAINING_WEIGHTS:-0.0:0.1:0.2:0.3:0.4:0.5:0.6:0.7:0.8:0.9:1.0}"
NUM_GRAPHS="${NUM_GRAPHS:-100}"

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
RELNET_SCRATCH="${RELNET_SCRATCH:-/home/ucabggs/Scratch}"
DENSITY_SOURCE="${DENSITY_SOURCE:-${SCRIPT_DIR}}"
LOG_DIR="${LOG_DIR:-${RELNET_SCRATCH}/relnet_preference_density_logs}"
SUBMISSION_LOG="${DENSITY_SOURCE}/density_submissions.tsv"

INITIAL_EDGE_COUNTS=(19 29 38 48 57 67)
MODEL_SEEDS=(0 42 84 126 168)

mkdir -p "$LOG_DIR"

for initial_edges in "${INITIAL_EDGE_COUNTS[@]}"
do
  m_tag=$(printf '%03d' "$initial_edges")
  calibration_variables="GRAPH_N=${GRAPH_N},INITIAL_EDGES=${initial_edges},EDGE_PERCENTAGE=${EDGE_PERCENTAGE},EDGE_BUDGET=${EDGE_BUDGET},NUM_GRAPHS=${NUM_GRAPHS}"
  calibration_output=$(qsub -terse \
    -N "c20m${m_tag}" \
    -o "$LOG_DIR" \
    -j y \
    -v "$calibration_variables" \
    "${DENSITY_SOURCE}/relnet_cpu_density_calibration.sh")
  calibration_job_id="${calibration_output%%.*}"
  printf '%s\t%s\tcalibration\tm=%s\n' \
    "$(date --iso-8601=seconds)" "$calibration_job_id" "$initial_edges" \
    >> "$SUBMISSION_LOG"
  echo "Submitted calibration m=${initial_edges}: ${calibration_job_id}"

  for model_seed in "${MODEL_SEEDS[@]}"
  do
    run_number=$((model_seed / 42))
    training_variables="GRAPH_N=${GRAPH_N},INITIAL_EDGES=${initial_edges},EDGE_PERCENTAGE=${EDGE_PERCENTAGE},EDGE_BUDGET=${EDGE_BUDGET},NETWORK_GENERATOR=random_network,MODEL_SEED=${model_seed},AGENT_BUDGET=${AGENT_BUDGET},VALIDATION_INTERVAL=${VALIDATION_INTERVAL},EXPERIMENT_VARIANT=${EXPERIMENT_VARIANT},TRAINING_WEIGHTS=${TRAINING_WEIGHTS}"
    training_output=$(qsub -terse \
      -N "d20m${m_tag}s${run_number}" \
      -hold_jid "$calibration_job_id" \
      -o "$LOG_DIR" \
      -j y \
      -v "$training_variables" \
      "${DENSITY_SOURCE}/relnet_cpu_preference_conditioned.sh")
    training_job_id="${training_output%%.*}"
    printf '%s\t%s\ttraining\tm=%s\tseed=%s\thold=%s\n' \
      "$(date --iso-8601=seconds)" "$training_job_id" \
      "$initial_edges" "$model_seed" "$calibration_job_id" \
      >> "$SUBMISSION_LOG"
    echo "Submitted training m=${initial_edges} seed=${model_seed}: ${training_job_id}"
  done
done

echo "Submitted 6 calibration jobs and 30 held training jobs."
echo "Submission record: $SUBMISSION_LOG"
