#!/bin/bash

set -euo pipefail

if [ "$#" -ne 5 ]
then
  echo "Usage: MODEL_SEED=SEED $0 N EDGE_PERCENTAGE L WEIGHT_INDEX GENERATOR"
  exit 2
fi

GRAPH_N="$1"
EDGE_PERCENTAGE="$2"
EDGE_BUDGET="$3"
WEIGHT_INDEX="$4"
NETWORK_GENERATOR="$5"
MODEL_SEED="${MODEL_SEED:-0}"
HOLD_JID="${HOLD_JID:-}"

case "$MODEL_SEED" in
  0|42|84|126|168) ;;
  *) echo "MODEL_SEED must be one of 0, 42, 84, 126 or 168"; exit 2 ;;
esac
case "$WEIGHT_INDEX" in
  0|1|2|3|4|5|6|7|8|9|10) ;;
  *) echo "WEIGHT_INDEX must be an integer from 0 to 10"; exit 2 ;;
esac
case "$NETWORK_GENERATOR" in
  random_network) GENERATOR_TAG=ER ;;
  barabasi_albert) GENERATOR_TAG=BA ;;
  *) echo "Unknown generator $NETWORK_GENERATOR"; exit 2 ;;
esac

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
WEIGHT_TAG=$(printf "%02d" "$WEIGHT_INDEX")
RUN_NUMBER=$((MODEL_SEED / 42))
JOB_NAME="r${GRAPH_N}L${EDGE_BUDGET}w${WEIGHT_TAG}${GENERATOR_TAG}m${RUN_NUMBER}"
VARIABLES="GRAPH_N=${GRAPH_N},EDGE_PERCENTAGE=${EDGE_PERCENTAGE},EDGE_BUDGET=${EDGE_BUDGET},WEIGHT_INDEX=${WEIGHT_INDEX},NETWORK_GENERATOR=${NETWORK_GENERATOR},MODEL_SEED=${MODEL_SEED}"

if [ -n "$HOLD_JID" ]
then
  qsub -N "$JOB_NAME" -hold_jid "$HOLD_JID" -v "$VARIABLES" \
    "$SCRIPT_DIR/relnet_cpu_single_task.sh"
else
  qsub -N "$JOB_NAME" -v "$VARIABLES" \
    "$SCRIPT_DIR/relnet_cpu_single_task.sh"
fi
