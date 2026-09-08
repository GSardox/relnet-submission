#!/bin/bash -l

#$ -N relnet_pref_ms
#$ -l mem=8G
#$ -l tmpfs=10G
#$ -l h_rt=72:00:00
#$ -wd /home/ucabggs/Scratch/relnet
#$ -j y

set -euo pipefail

module load apptainer

RELNET_SCRATCH="${RELNET_SCRATCH:-/home/ucabggs/Scratch}"
BASE_SOURCE="${RELNET_SOURCE_DIR:-${RELNET_SCRATCH}/relnet}"
STANDARD_DATA="${STANDARD_DATA:-${RELNET_SCRATCH}/experiment_data/relnet_preference_s2v}"
DENSITY_DATA="${DENSITY_DATA:-${RELNET_SCRATCH}/experiment_data/relnet_preference_density}"
STANDARD_LOG_DIR="${STANDARD_LOG_DIR:-${RELNET_SCRATCH}/relnet_preference_s2v_logs}"
DENSITY_LOG_DIR="${DENSITY_LOG_DIR:-${RELNET_SCRATCH}/relnet_preference_density_logs}"
IMAGE="${RELNET_CPU_IMAGE:-${RELNET_SCRATCH}/containers/relnet-worker-cpu.sif}"

set -a
source "${BASE_SOURCE}/relnet.env"
set +a

WHICH=synth
GRAPH_N="${GRAPH_N:?submit with GRAPH_N}"
EDGE_PERCENTAGE="${EDGE_PERCENTAGE:?submit with EDGE_PERCENTAGE}"
EDGE_BUDGET="${EDGE_BUDGET:?submit with EDGE_BUDGET}"
NETWORK_GENERATOR="${NETWORK_GENERATOR:-random_network}"
MODEL_SEED="${MODEL_SEED:?submit with MODEL_SEED}"
AGENT_BUDGET="${AGENT_BUDGET:-400000}"
TRAINING_WEIGHTS="${TRAINING_WEIGHTS:-0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0}"
VALIDATION_INTERVAL="${VALIDATION_INTERVAL:-5000}"
EXPERIMENT_VARIANT="${EXPERIMENT_VARIANT:-s2v_full400k}"
INITIAL_EDGES="${INITIAL_EDGES:-}"

case "$MODEL_SEED" in
  0|42|84|126|168) ;;
  *) echo "MODEL_SEED must be one of 0, 42, 84, 126 or 168"; exit 2 ;;
esac

if [[ ! "$EXPERIMENT_VARIANT" =~ ^[A-Za-z0-9_]+$ ]]
then
  echo "EXPERIMENT_VARIANT may contain only letters, numbers and underscores"
  exit 2
fi

for value in "$GRAPH_N" "$EDGE_BUDGET" "$MODEL_SEED" "$AGENT_BUDGET" "$VALIDATION_INTERVAL"
do
  if [[ ! "$value" =~ ^[0-9]+$ ]]
  then
    echo "graph, budget, seed and interval arguments must be integers"
    exit 2
  fi
done

if [[ ! "$EDGE_PERCENTAGE" =~ ^[0-9]+([.][0-9]+)?$ ]]
then
  echo "EDGE_PERCENTAGE must be a non-negative decimal number"
  exit 2
fi

if [[ ! "$TRAINING_WEIGHTS" =~ ^[0-9.,:]+$ ]]
then
  echo "TRAINING_WEIGHTS contains invalid characters"
  exit 2
fi

if [ "$AGENT_BUDGET" -le 0 ] || [ "$AGENT_BUDGET" -gt 400000 ]
then
  echo "AGENT_BUDGET must lie in [1, 400000]"
  exit 2
fi

if [ "$VALIDATION_INTERVAL" -le 0 ]
then
  echo "VALIDATION_INTERVAL must be positive"
  exit 2
fi

DENSITY_MODE=0
if [ -n "$INITIAL_EDGES" ]
then
  DENSITY_MODE=1
  if [[ ! "$INITIAL_EDGES" =~ ^[0-9]+$ ]]
  then
    echo "INITIAL_EDGES must be an integer"
    exit 2
  fi
  WHICH=synth
  NETWORK_GENERATOR=random_network
  PREFERENCE_DATA="$DENSITY_DATA"
  LOG_DIR="$DENSITY_LOG_DIR"
  M_TAG=$(printf '%03d' "$INITIAL_EDGES")
  EXPERIMENT_ID="relnet_cpu_pref_${EXPERIMENT_VARIANT}_n${GRAPH_N}_m${M_TAG}_L${EDGE_BUDGET}_s5"
  NORMALIZATION_PATH="${DENSITY_DATA}/normalization/er_n${GRAPH_N}_m${M_TAG}_L${EDGE_BUDGET}.json"
  if [ ! -s "$NORMALIZATION_PATH" ]
  then
    echo "Missing completed normalization: $NORMALIZATION_PATH"
    exit 3
  fi
else
  PREFERENCE_DATA="$STANDARD_DATA"
  LOG_DIR="$STANDARD_LOG_DIR"
  case "$NETWORK_GENERATOR" in
    random_network|barabasi_albert) ;;
    *) echo "NETWORK_GENERATOR is invalid"; exit 2 ;;
  esac
  EXPERIMENT_ID="relnet_cpu_pref_${EXPERIMENT_VARIANT}_n${GRAPH_N}_L${EDGE_BUDGET}_s1exact"
fi

TASK_LOG="${LOG_DIR}/preference_${JOB_ID}.log"
MODEL_PATH="${PREFERENCE_DATA}/${EXPERIMENT_ID}/models/checkpoints/rnet_dqn_pref-combined_linear-${NETWORK_GENERATOR}-${MODEL_SEED}-0/rnet_dqn_pref_agent.model"
MANIFEST_PATH="${PREFERENCE_DATA}/${EXPERIMENT_ID}/complete_preference_${NETWORK_GENERATOR}_seed${MODEL_SEED}.json"
LOCK_DIR="${PREFERENCE_DATA}/${EXPERIMENT_ID}/.running_preference_${NETWORK_GENERATOR}_seed${MODEL_SEED}"

mkdir -p "${PREFERENCE_DATA}/${EXPERIMENT_ID}" "$LOG_DIR"

if [ -e "$MODEL_PATH" ] || [ -e "$MANIFEST_PATH" ]
then
  echo "Refusing to overwrite an existing preference-conditioned model."
  echo "$MODEL_PATH"
  echo "$MANIFEST_PATH"
  exit 3
fi

if ! mkdir "$LOCK_DIR"
then
  echo "Refusing duplicate preference-conditioned run: $LOCK_DIR"
  exit 3
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

export APPTAINERENV_RELNET_DEVICE_PLACEMENT=CPU
export APPTAINERENV_RN_LABORER_PW="$RN_LABORER_PW"
export APPTAINERENV_PYTHONPATH=/relnet
export APPTAINERENV_HOSTNAME="relnet-pref-${JOB_ID}"
export APPTAINERENV_PYTHONDONTWRITEBYTECODE=1
export APPTAINERENV_OMP_NUM_THREADS=1
export APPTAINERENV_MKL_NUM_THREADS=1
export APPTAINERENV_OPENBLAS_NUM_THREADS=1
export APPTAINERENV_NUMEXPR_NUM_THREADS=1

echo "job_id=$JOB_ID"
echo "experiment_id=$EXPERIMENT_ID"
echo "network_generator=$NETWORK_GENERATOR"
echo "initial_edges=${INITIAL_EDGES:-standard}"
echo "model_seed=$MODEL_SEED"
echo "training_weights=$TRAINING_WEIGHTS"
echo "agent_budget=$AGENT_BUDGET"
echo "validation_interval=$VALIDATION_INTERVAL"

PYTHON_ARGS=(
  --which "$WHICH"
  --n "$GRAPH_N"
  --edge_percentage "$EDGE_PERCENTAGE"
  --edge_budget "$EDGE_BUDGET"
  --network_generator "$NETWORK_GENERATOR"
  --agent_budget "$AGENT_BUDGET"
  --model_seed "$MODEL_SEED"
  --experiment_id "$EXPERIMENT_ID"
  --parent_dir /preference_data
  --weights "$TRAINING_WEIGHTS"
  --validation_check_interval "$VALIDATION_INTERVAL"
)

if [ "$DENSITY_MODE" -eq 1 ]
then
  PYTHON_ARGS+=(
    --initial_edges "$INITIAL_EDGES"
    --normalization_file "/preference_data/normalization/$(basename "$NORMALIZATION_PATH")"
  )
fi

/usr/bin/time --verbose \
apptainer exec --compat --pwd /relnet \
  --bind "$BASE_SOURCE:/relnet:ro" \
  --bind "$PREFERENCE_DATA:/preference_data" \
  "$IMAGE" \
  /opt/conda/envs/ucfadar-relnet/bin/python -B -u \
  /relnet/run_preference_training.py \
  "${PYTHON_ARGS[@]}" \
  2>&1 | tee "$TASK_LOG"

if [ ! -s "$MODEL_PATH" ] || [ ! -s "$MANIFEST_PATH" ]
then
  echo "Preference training exited but completion files are missing."
  exit 1
fi

echo "Verified model: $MODEL_PATH"
echo "Verified manifest: $MANIFEST_PATH"
