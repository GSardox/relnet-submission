#!/bin/bash -l

#$ -N relnet_den_cal
#$ -l mem=4G
#$ -l tmpfs=10G
#$ -l h_rt=72:00:00
#$ -wd /home/ucabggs/Scratch/relnet
#$ -j y

set -euo pipefail

module load apptainer

RELNET_SCRATCH="${RELNET_SCRATCH:-/home/ucabggs/Scratch}"
BASE_SOURCE="${RELNET_SOURCE_DIR:-${RELNET_SCRATCH}/relnet}"
DENSITY_DATA="${DENSITY_DATA:-${RELNET_SCRATCH}/experiment_data/relnet_preference_density}"
LOG_DIR="${LOG_DIR:-${RELNET_SCRATCH}/relnet_preference_density_logs}"
IMAGE="${RELNET_CPU_IMAGE:-${RELNET_SCRATCH}/containers/relnet-worker-cpu.sif}"

set -a
source "${BASE_SOURCE}/relnet.env"
set +a

GRAPH_N="${GRAPH_N:?submit with GRAPH_N}"
INITIAL_EDGES="${INITIAL_EDGES:?submit with INITIAL_EDGES}"
EDGE_PERCENTAGE="${EDGE_PERCENTAGE:?submit with EDGE_PERCENTAGE}"
EDGE_BUDGET="${EDGE_BUDGET:?submit with EDGE_BUDGET}"
NUM_GRAPHS="${NUM_GRAPHS:-100}"

for value in "$GRAPH_N" "$INITIAL_EDGES" "$EDGE_BUDGET" "$NUM_GRAPHS"
do
  if [[ ! "$value" =~ ^[0-9]+$ ]]
  then
    echo "graph, edge and count arguments must be integers"
    exit 2
  fi
done

if [[ ! "$EDGE_PERCENTAGE" =~ ^[0-9]+([.][0-9]+)?$ ]]
then
  echo "EDGE_PERCENTAGE must be a non-negative decimal number"
  exit 2
fi

NORMALIZATION_DIR="${DENSITY_DATA}/normalization"
NORMALIZATION_PATH="${NORMALIZATION_DIR}/er_n${GRAPH_N}_m$(printf '%03d' "$INITIAL_EDGES")_L${EDGE_BUDGET}.json"
LOCK_DIR="${NORMALIZATION_PATH}.running"
TASK_LOG="${LOG_DIR}/calibration_${JOB_ID}.log"

mkdir -p "$NORMALIZATION_DIR" "$LOG_DIR"

if [ -e "$NORMALIZATION_PATH" ]
then
  echo "Refusing to overwrite existing normalization: $NORMALIZATION_PATH"
  exit 3
fi

if ! mkdir "$LOCK_DIR"
then
  echo "Refusing duplicate calibration: $LOCK_DIR"
  exit 3
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

export APPTAINERENV_RELNET_DEVICE_PLACEMENT=CPU
export APPTAINERENV_RN_LABORER_PW="$RN_LABORER_PW"
export APPTAINERENV_PYTHONPATH=/relnet
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
export APPTAINERENV_OMP_NUM_THREADS=1
export APPTAINERENV_MKL_NUM_THREADS=1
export APPTAINERENV_OPENBLAS_NUM_THREADS=1
export APPTAINERENV_NUMEXPR_NUM_THREADS=1

echo "job_id=$JOB_ID"
echo "n=$GRAPH_N"
echo "initial_edges=$INITIAL_EDGES"
echo "edge_budget=$EDGE_BUDGET"
echo "num_graphs=$NUM_GRAPHS"

/usr/bin/time --verbose \
apptainer exec --compat --pwd /relnet \
  --bind "$BASE_SOURCE:/relnet:ro" \
  --bind "$DENSITY_DATA:/density_data" \
  "$IMAGE" \
  /opt/conda/envs/ucfadar-relnet/bin/python -B -u \
  /relnet/tools/calibrate_normalization.py density \
  --n "$GRAPH_N" \
  --initial_edges "$INITIAL_EDGES" \
  --edge_percentage "$EDGE_PERCENTAGE" \
  --edge_budget "$EDGE_BUDGET" \
  --num_graphs "$NUM_GRAPHS" \
  --num_mc_sims "$((GRAPH_N * 2))" \
  --random_seed 42 \
  --output "/density_data/normalization/$(basename "$NORMALIZATION_PATH")" \
  2>&1 | tee "$TASK_LOG"

if [ ! -s "$NORMALIZATION_PATH" ]
then
  echo "Calibration exited but normalization output is missing."
  exit 1
fi

echo "Verified normalization: $NORMALIZATION_PATH"
