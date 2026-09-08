#!/bin/bash -l

set -e
module load apptainer

DEVICE="${1:?use CPU or GPU}"
case "$DEVICE" in
  CPU)
    IMAGE_NAME=relnet-worker-cpu.sif
    WORKER_NAME=cpu0
    CONCURRENCY=22
    APPTAINER_GPU=()
    DEFAULT_VARIANT=relnet_cpu_v2
    ;;
  GPU)
    IMAGE_NAME=relnet-worker-gpu.sif
    WORKER_NAME=gpu0
    CONCURRENCY=4
    APPTAINER_GPU=(--nv)
    DEFAULT_VARIANT=relnet_cpu_final
    ;;
  *) echo "Device must be CPU or GPU"; exit 2 ;;
esac

RELNET_SCRATCH="${RELNET_SCRATCH:-/home/ucabggs/Scratch}"
source "${RELNET_SCRATCH}/relnet/relnet.env"

export RN_SOURCE_DIR="${RN_SOURCE_DIR:-${RELNET_SCRATCH}/relnet}"
export RN_APP_DATA_DIR="${RN_APP_DATA_DIR:-${RELNET_SCRATCH}/app_data/relnet_${JOB_ID}}"
export RN_EXPERIMENT_DATA_DIR="${RN_EXPERIMENT_DATA_DIR:-${RELNET_SCRATCH}/experiment_data/relnet}"
export RN_GID="$(id -g)"
export RN_GNAME="$(id -gn)"
export APPTAINERENV_RN_GID="$RN_GID"
export APPTAINERENV_RN_GNAME="$RN_GNAME"
export APPTAINERENV_RN_LABORER_PW="$RN_LABORER_PW"
export APPTAINERENV_RN_ADMIN_PW="$RN_ADMIN_PW"
export APPTAINERENV_OMP_NUM_THREADS=1
export APPTAINERENV_MKL_NUM_THREADS=1
export APPTAINERENV_OPENBLAS_NUM_THREADS=1
export APPTAINERENV_NUMEXPR_NUM_THREADS=1

CODE="$RN_SOURCE_DIR"
DATA="$RN_EXPERIMENT_DATA_DIR"
APP="$RN_APP_DATA_DIR"
RABBIT="${TMPDIR:-$APP}/rn_rabbitmq"
MONGO="${TMPDIR:-$APP}/rn_mongodb"
IMAGES="${RELNET_CONTAINER_DIR:-${RELNET_SCRATCH}/containers}"
LOGS="${RELNET_LOG_DIR:-${RELNET_SCRATCH}/relnet_job_logs}"
HOSTS_FILE="${RELNET_SCRATCH}/relnet_hosts"

pkill -9 -f 'starter-suid' 2>/dev/null || true
pkill -9 -f beam.smp 2>/dev/null || true
pkill -9 -f epmd 2>/dev/null || true
pkill -9 -f rabbitmq 2>/dev/null || true
pkill -9 -f mongod 2>/dev/null || true
sleep 5

rm -rf "$APP/rabbitmq"
mkdir -p "$DATA" "$MONGO/data" "$APP/mongo_dump" \
  "$RABBIT/data" "$RABBIT/log" "$APP/flower" "$LOGS"
printf '127.0.0.1   localhost\n127.0.0.1   relnet-manager relnet-worker-cpu relnet-worker-gpu\n' > "$HOSTS_FILE"

cleanup() {
  pkill -9 -f 'starter-suid' 2>/dev/null || true
  pkill -9 -f beam.smp 2>/dev/null || true
  pkill -9 -f epmd 2>/dev/null || true
}
trap cleanup EXIT INT TERM

apptainer run --compat \
  --bind "$HOSTS_FILE:/etc/hosts" \
  --bind "$MONGO/data:/data/db" \
  "$IMAGES/mongo-4.1.sif" > "$LOGS/mongodb_${JOB_ID}.log" 2>&1 &

APPTAINERENV_HOSTNAME=relnet-manager apptainer run --compat \
  --bind "$HOSTS_FILE:/etc/hosts" \
  --bind "$CODE:/relnet" \
  --bind "$DATA:/experiment_data" \
  --bind "$RABBIT/data:/var/lib/rabbitmq" \
  --bind "$RABBIT/log:/var/log/rabbitmq" \
  --bind "$APP/flower:/flower" \
  "$IMAGES/relnet-manager.sif" > "$LOGS/manager_${JOB_ID}.log" 2>&1 &

echo 'Waiting for the RabbitMQ worker account...'
for i in $(seq 1 120)
do
  if apptainer exec --compat \
    --bind "$HOSTS_FILE:/etc/hosts" \
    --bind "$RABBIT/data:/var/lib/rabbitmq" \
    --bind "$RABBIT/log:/var/log/rabbitmq" \
    "$IMAGES/relnet-manager.sif" bash -lc \
    "export HOME=/var/lib/rabbitmq; export RABBITMQ_NODENAME=rabbit@localhost; rabbitmqctl -n rabbit@localhost list_users -q 2>/dev/null | grep -q relnetlaborer"
  then
    echo "RabbitMQ ready after $((i * 5)) seconds."
    break
  fi
  sleep 5
done

(
  for attempt in $(seq 1 60)
  do
    echo "[$WORKER_NAME] launch attempt $attempt"
    APPTAINERENV_HOSTNAME="relnet-worker-${DEVICE,,}" \
    APPTAINERENV_RELNET_DEVICE_PLACEMENT="$DEVICE" \
    apptainer exec --compat "${APPTAINER_GPU[@]}" --pwd /relnet \
      --bind "$HOSTS_FILE:/etc/hosts" \
      --bind "$DATA:/experiment_data" \
      --bind "$CODE:/relnet" \
      "$IMAGES/$IMAGE_NAME" bash -lc \
      "source activate ucfadar-relnet; cd /relnet; celery -A tasks worker -Ofair --loglevel=info --without-gossip --without-mingle --hostname=${WORKER_NAME}@%h --concurrency=${CONCURRENCY}" \
      > "$LOGS/${WORKER_NAME}_${JOB_ID}.log" 2>&1 || true
    sleep 20
  done
) &
sleep 20

GRAPH_N="${GRAPH_N:-10}"
EDGE_PERCENTAGE="${EDGE_PERCENTAGE:-1}"
EDGE_BUDGET="${EDGE_BUDGET:-1}"
EXPERIMENT_VARIANT="${EXPERIMENT_VARIANT:-$DEFAULT_VARIANT}"
PIDS=()

for weight_index in $(seq 0 10)
do
  weight=$(awk -v k="$weight_index" 'BEGIN {printf "%.1f", k/10}')
  tag=$(printf '%02d' "$weight_index")
  experiment_id="${EXPERIMENT_VARIANT}_n${GRAPH_N}_L${EDGE_BUDGET}_w${tag}_s1exact"
  [ "$DEVICE" = GPU ] && experiment_id="${experiment_id}_gpu"
  echo "Launching weight $weight: $experiment_id"
  APPTAINERENV_HOSTNAME=relnet-manager apptainer exec --compat --pwd /relnet \
    --bind "$HOSTS_FILE:/etc/hosts" --bind "$CODE:/relnet" \
    --bind "$DATA:/experiment_data" "$IMAGES/relnet-manager.sif" bash -lc \
    "source activate ucfadar-relnet; cd /relnet; python -u run_experiments.py --which synth --n ${GRAPH_N} --experiment_part hyperopt --edge_percentage ${EDGE_PERCENTAGE} --weight ${weight} --experiment_id ${experiment_id} --parent_dir /experiment_data --run_num_start 0 --run_num_end 0 --force_insert_details" \
    > "$LOGS/experiment_w${tag}_${JOB_ID}.log" 2>&1 &
  PIDS+=("$!")
done

run_status=0
for process_id in "${PIDS[@]}"
do
  wait "$process_id" || run_status=1
done

echo 'Dumping MongoDB results to Scratch...'
apptainer exec --compat \
  --bind "$MONGO/data:/data/db" --bind "$APP/mongo_dump:/dump" \
  "$IMAGES/mongo-4.1.sif" bash -lc \
  'mongod --dbpath /data/db --fork --logpath /tmp/md.log --bind_ip 127.0.0.1 && sleep 5 && mongodump --db relnet_jobs --out /dump && mongod --dbpath /data/db --shutdown'

echo "Results: $DATA"
echo "Logs: $LOGS/*_${JOB_ID}.log"
exit "$run_status"
