
TEST_TYPE=${1:-allreduce}
ADDITIONAL_LD_LIBRARY_PATH=${2:-/usr/local/cuda-12.4/lib}
DATA_PATTERN=${3:-0x0}
SLURM_JOB_NUM_NODES=${3:-2}

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
        TEST_BINARY="${CUDA_TEST_DIR}/all_reduce_perf"
        ;;
esac
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
    -x NCCL_TESTS_SPLIT_MASK=${DATA_PATTERN} \
	--mca pml ^ucx \
	--mca btl tcp,self \
	--mca btl_tcp_if_exclude lo,docker0,veth_def_agent \
    --mca rmaps seq \
    --hostfile topo_sorted_hostnames.txt \
	--bind-to none ${TEST_BINARY} -b 8 -e 16G -f 2 -g 1 -c 1 -n 100 