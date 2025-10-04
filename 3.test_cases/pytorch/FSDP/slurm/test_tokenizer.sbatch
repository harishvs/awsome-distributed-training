#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=test-tokenizer
#SBATCH --output=logs/test-tokenizer_%j.out
#SBATCH --error=logs/test-tokenizer_%j.err
#SBATCH --time=00:05:00

set -ex

export CONTAINER_IMAGE=$(pwd)/pytorch-fsdp.sqsh
export FSX_MOUNT=/fsxl:/fsxl

srun --container-image $CONTAINER_IMAGE \
     --container-mounts $FSX_MOUNT \
     python3 test_tokenizer.py
