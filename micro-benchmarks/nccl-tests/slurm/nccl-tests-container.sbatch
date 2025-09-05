#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --job-name=nccl-tests-container # name of your job
#SBATCH --nodes=16 # number of nodes to use, 2 p5en.48xlarge = 16 H200 GPUs
#SBATCH --ntasks-per-node 8 # Number of GPU per node (8 H200 per p5en.48xlarge)
#SBATCH --constraint=p5en.48xlarge # specify instance type
###SBATCH --gpus-per-node=8 # number of GPU we reserve. Uncomment for AWS ParallelCluster
#SBATCH --output %x_%j.out
#SBATCH --error %x_%j.err
#SBATCH --exclusive
#SBATCH --wait-all-nodes=1

### Disable hyperthreading by setting the tasks per core to 1
#SBATCH --ntasks-per-core=1

###########################
###### User Variables #####
###########################

# Supported NCCL collective operations:
# - allreduce     : AllReduce collective (default)
# - allgather     : AllGather collective  
# - reducescatter : ReduceScatter collective
# - alltoall      : AllToAll collective

TEST_TYPE=${1:-allreduce}
APPS_PATH=${2:-/fsx}
DATA_PATTERN=${3:-0x0}

# default variables for Enroot
: "${NCCL_TESTS_PATH:=/opt/nccl-tests/build}"
: "${IMAGE:=$APPS_PATH/nccl-tests.sqsh}"

# Set binary path based on test type
case ${TEST_TYPE} in
    allreduce)
        TEST_BINARY="${NCCL_TESTS_PATH}/all_reduce_perf"
        ;;
    allgather)
        TEST_BINARY="${NCCL_TESTS_PATH}/all_gather_perf"
        ;;
    reducescatter)
        TEST_BINARY="${NCCL_TESTS_PATH}/reduce_scatter_perf"
        ;;
    alltoall)
        TEST_BINARY="${NCCL_TESTS_PATH}/alltoall_perf"
        ;;
    *)
        echo "Error: Unsupported test type '${TEST_TYPE}'"
        echo "Supported types: allreduce, allgather, reducescatter, alltoall"
        exit 1
        ;;
esac

echo "Running NCCL ${TEST_TYPE} test with data pattern ${DATA_PATTERN}"

export NCCL_TESTS_SPLIT_MASK=${DATA_PATTERN}

## Set libfabric flags to use EFA
export FI_PROVIDER=efa
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4d
export FI_EFA_FORK_SAFE=1

## Set this flag for debugging EFA
#export FI_LOG_LEVEL=warn

## NCCL Environment variables
# export NCCL_DEBUG=INFO

### Increase the send queue depth and can turn NCCL communications into non-blocking.
### https://www.usenix.org/system/files/atc23-choi.pdf
export NCCL_BUFFSIZE=8388608
### Improve performance by increasing buffer size for Send/Recv, Gather, Scatter and Alltoall communications
### https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html
export NCCL_P2P_NET_CHUNKSIZE=524288

### Improve performance for AllReduce by selecting specific protocol and algorithm for specific
### message size and number of ranks.
### More information https://github.com/aws/aws-ofi-nccl/wiki/Algorithm-and-Protocol-Tuner-for-AWS.
export NCCL_TUNER_PLUGIN=/opt/aws-ofi-nccl/install/lib/libnccl-ofi-tuner.so


declare -a ARGS=(
    --container-image $IMAGE
)

#Get Hostname and Instance IDs
mpirun -N 1 bash -c 'echo $(hostname): $(cat /sys/devices/virtual/dmi/id/board_asset_tag | tr -d " ")'

# Run NCCL test with configurable data pattern
srun "${ARGS[@]}" --mpi=pmix --cpu-bind=none $TEST_BINARY -b 8 -e 16G -f 2 -g 1 -c 1 -n 100
