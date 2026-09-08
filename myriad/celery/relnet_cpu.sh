#!/bin/bash -l

#$ -N relnet_cprod
#$ -pe smp 22
#$ -l mem=4G
#$ -l tmpfs=20G
#$ -l h_rt=48:00:00
#$ -wd /home/ucabggs/Scratch
#$ -j y

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
exec "$SCRIPT_DIR/run_celery_job.sh" CPU
