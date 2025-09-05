#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

#SBATCH --job-name=nccl-tests-0x0 # name of your job
#SBATCH --nodes=2 # number of nodes to use, 2 p5en.48xlarge = 16 H200 GPUs
#SBATCH --ntasks-per-node 8 # Number of GPU per node (8 H200 per p5en.48xlarge)
#SBATCH --constraint=p5en.48xlarge # specify instance type
#SBATCH --output %x_%j.out
#SBATCH --error %x_%j.err
#SBATCH --exclusive

set -ex

# This script is designed to run by default on the Deep Learning AMI, Ubuntu 20.04
# See https://aws.amazon.com/releasenotes/aws-deep-learning-base-gpu-ami-ubuntu-20-04/

# Supported NCCL collective operations:
# - allreduce     : AllReduce collective (default)
# - allgather     : AllGather collective  
# - reducescatter : ReduceScatter collective
# - alltoall      : AllToAll collective

TEST_TYPE=${1:-allreduce}
ADDITIONAL_LD_LIBRARY_PATH=${2:-/usr/local/cuda-12.4/lib}
DATA_PATTERN=${3:-0x0}

# Set binary path based on test type
CUDA_TEST_DIR="/usr/local/cuda-12.4/efa/test-cuda-12.4"
case ${TEST_TYPE} in
    allreduce)
        TEST_BINARY="${CUDA_TEST_DIR}/all_reduce_perf"
        ;;
    allgather)
        TEST_BINARY="${CUDA_TEST_DIR}/all_gather_perf"
        ;;
    reducescatter)
        TEST_BINARY="${CUDA_TEST_DIR}/reduce_scatter_perf"
        ;;
    alltoall)
        TEST_BINARY="${CUDA_TEST_DIR}/alltoall_perf"
        ;;
    *)
        echo "Error: Unsupported test type '${TEST_TYPE}'"
        echo "Supported types: allreduce, allgather, reducescatter, alltoall"
        exit 1
        ;;
esac

echo "Running NCCL ${TEST_TYPE} test with data pattern ${DATA_PATTERN}"

# Get Hostname to Instance ID mapping
mpirun -N 1 bash -c 'echo $(hostname) ➡️ $(cat /sys/devices/virtual/dmi/id/board_asset_tag | tr -d " ")'


### NCCL_BUFFSIZE increase the send queue depth and can turn NCCL communications into non-blocking.
### https://www.usenix.org/system/files/atc23-choi.pdf

### NCCL_P2P_NET_CHUNKSIZE Improve performance by increasing buffer size for Send/Recv, Gather, Scatter and Alltoall communications
### https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html

### Improve performance for AllReduce by selecting specific protocol and algorithm for specific
### message size and number of ranks.
### More information https://github.com/aws/aws-ofi-nccl/wiki/Algorithm-and-Protocol-Tuner-for-AWS.

# run NCCL test with configurable data pattern (default: 0x0)
mpirun -n $((8 * SLURM_JOB_NUM_NODES)) -N 8 \
        -x FI_PROVIDER=efa \
	-x FI_EFA_USE_DEVICE_RDMA=1  \
	-x FI_EFA_FORK_SAFE=1 \
	-x LD_LIBRARY_PATH=$ADDITIONAL_LD_LIBRARY_PATH:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:/opt/amazon/ofi-nccl/lib:/usr/local/lib:/usr/lib:$LD_LIBRARY_PATH \
	-x NCCL_DEBUG=INFO \
	-x NCCL_SOCKET_IFNAME=^docker,lo,veth \
	-x NCCL_BUFFSIZE=8388608 \
	-x NCCL_P2P_NET_CHUNKSIZE=524288 \
	-x NCCL_TUNER_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-ofi-tuner.so \
	--mca pml ^ucx \
	--mca btl tcp,self \
	--mca btl_tcp_if_exclude lo,docker0,veth_def_agent \
	--bind-to none ${TEST_BINARY} -b 8 -e 16G -f 2 -g 1 -c 1 -n 100 
    # -d ${DATA_PATTERN}

