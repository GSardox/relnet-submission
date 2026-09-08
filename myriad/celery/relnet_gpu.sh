#!/bin/bash -l

#$ -N relnet_gprod
#$ -pe smp 22
#$ -l gpu=1
#$ -l mem=1.5G
#$ -l tmpfs=40G
#$ -l h_rt=48:00:00
#$ -wd /home/ucabggs/Scratch
#$ -j y

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
exec "$SCRIPT_DIR/run_celery_job.sh" GPU
