#!/bin/bash -l

#$ -N relnet_one
#$ -l mem=4G
#$ -l tmpfs=10G
#$ -l h_rt=72:00:00
#$ -wd /home/ucabggs/Scratch
#$ -j y

set -euo pipefail

module load apptainer

RELNET_SCRATCH="${RELNET_SCRATCH:-/home/ucabggs/Scratch}"
set -a
source "${RELNET_SCRATCH}/relnet/relnet.env"
set +a

GRAPH_N="${GRAPH_N:?submit with GRAPH_N}"
EDGE_PERCENTAGE="${EDGE_PERCENTAGE:?submit with EDGE_PERCENTAGE}"
EDGE_BUDGET="${EDGE_BUDGET:?submit with EDGE_BUDGET}"
WEIGHT_INDEX="${WEIGHT_INDEX:?submit with WEIGHT_INDEX from 0 to 10}"
NETWORK_GENERATOR="${NETWORK_GENERATOR:?submit with NETWORK_GENERATOR}"
MODEL_SEED="${MODEL_SEED:-0}"

case "$MODEL_SEED" in
  0|42|84|126|168) ;;
  *) echo "MODEL_SEED must be one of 0, 42, 84, 126 or 168"; exit 2 ;;
esac

case "$NETWORK_GENERATOR" in
  random_network|barabasi_albert) ;;
  *) echo "NETWORK_GENERATOR must be random_network or barabasi_albert"; exit 2 ;;
esac

case "$WEIGHT_INDEX" in
  0|1|2|3|4|5|6|7|8|9|10) ;;
  *) echo "WEIGHT_INDEX must be an integer from 0 to 10"; exit 2 ;;
esac

WEIGHT=$(awk -v k="$WEIGHT_INDEX" 'BEGIN {printf "%.1f", k/10}')
WEIGHT_TAG=$(printf "%02d" "$WEIGHT_INDEX")
AGENT_BUDGET=$((20000 * EDGE_BUDGET))
EXPERIMENT_ID="relnet_cpu_v4_n${GRAPH_N}_L${EDGE_BUDGET}_w${WEIGHT_TAG}_s1exact"
SOURCE_DIR="${RELNET_SOURCE_DIR:-${RELNET_SCRATCH}/relnet}"
DATA_DIR="${RELNET_DATA_DIR:-${RELNET_SCRATCH}/experiment_data/relnet}"
IMAGE="${RELNET_CPU_IMAGE:-${RELNET_SCRATCH}/containers/relnet-worker-cpu.sif}"
LOG_DIR="${RELNET_LOG_DIR:-${RELNET_SCRATCH}/relnet_job_logs}"
TASK_LOG="${LOG_DIR}/single_${JOB_ID}.log"
MODEL_PATH="${DATA_DIR}/${EXPERIMENT_ID}/models/checkpoints/rnet_dqn-combined_linear-${NETWORK_GENERATOR}-${MODEL_SEED}-0/rnet_dqn_agent.model"
if [ "$MODEL_SEED" -eq 0 ]
then
  MANIFEST_PATH="${DATA_DIR}/${EXPERIMENT_ID}/complete_${NETWORK_GENERATOR}.json"
else
  MANIFEST_PATH="${DATA_DIR}/${EXPERIMENT_ID}/complete_${NETWORK_GENERATOR}_seed${MODEL_SEED}.json"
fi
LOCK_DIR="${DATA_DIR}/${EXPERIMENT_ID}/.running_${NETWORK_GENERATOR}_seed${MODEL_SEED}"

mkdir -p "${DATA_DIR}/${EXPERIMENT_ID}" "$LOG_DIR"
if [ -e "$MODEL_PATH" ] || [ -e "$MANIFEST_PATH" ]
then
  echo "Refusing to overwrite $MODEL_PATH or $MANIFEST_PATH"
  exit 3
fi
if ! mkdir "$LOCK_DIR"
then
  echo "Refusing duplicate run at $LOCK_DIR"
  exit 3
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

export APPTAINERENV_RELNET_DEVICE_PLACEMENT=CPU
export APPTAINERENV_RN_LABORER_PW="$RN_LABORER_PW"
export APPTAINERENV_OMP_NUM_THREADS=1
export APPTAINERENV_MKL_NUM_THREADS=1
export APPTAINERENV_OPENBLAS_NUM_THREADS=1
export APPTAINERENV_NUMEXPR_NUM_THREADS=1

/usr/bin/time --verbose \
apptainer exec --compat --pwd /relnet \
  --bind "$SOURCE_DIR:/relnet" \
  --bind "$DATA_DIR:/experiment_data" \
  "$IMAGE" \
  bash -lc "export HOSTNAME='relnet-single-${JOB_ID}'; source activate ucfadar-relnet; cd /relnet; python -u run_single_training.py --which synth --n '${GRAPH_N}' --edge_percentage '${EDGE_PERCENTAGE}' --edge_budget '${EDGE_BUDGET}' --weight '${WEIGHT}' --network_generator '${NETWORK_GENERATOR}' --agent_budget '${AGENT_BUDGET}' --model_seed '${MODEL_SEED}' --experiment_id '${EXPERIMENT_ID}' --parent_dir /experiment_data" \
  2>&1 | tee "$TASK_LOG"

if [ ! -s "$MODEL_PATH" ] || [ ! -s "$MANIFEST_PATH" ]
then
  echo "Required output is missing"
  exit 1
fi
