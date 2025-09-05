# NCCL Performance Tests on AWS p5en.48xlarge

This directory contains SLURM batch scripts for running NCCL collective communication performance tests on AWS p5en.48xlarge instances with H200 GPUs.

## Available Scripts

- **`nccl-tests-ami.sbatch`**: Runs NCCL tests using binaries from Deep Learning AMI
- **`nccl-tests-container.sbatch`**: Runs NCCL tests using containerized environment

Both scripts run NCCL performance benchmarks across 2 nodes (16 H200 GPUs total) and support four key collective operations with configurable data patterns.

## Container Setup

To use the `nccl-tests-container.sbatch` script, you need to build and prepare the NCCL tests container.

### Building the Container

1. **Navigate to the container directory:**
   ```bash
   cd micro-benchmarks/nccl-tests/
   ```

2. **Create environment file (optional):**
   ```bash
   # Copy the example environment file
   cp slurm/.env.example .env
   
   # Edit to customize CUDA version and other parameters
   vim .env
   ```

3. **Build the container:**
   ```bash
   # Using default CUDA version (12.8.1)
   docker build -f nccl-tests.Dockerfile -t nccl-tests:latest .
   
   # Or with custom CUDA version
   docker build -f nccl-tests.Dockerfile \
     --build-arg CUDA_VERSION=12.4.1 \
     -t nccl-tests:cuda-12.4.1 .
   ```

4. **Convert to Enroot format:**
   ```bash
   # Create the squashfs image for Enroot
   enroot import -o /fsxl/nccl-tests.sqsh dockerd://nccl-tests:latest
   
   # Or with custom tag
   enroot import -o /fsxl/nccl-tests-cuda-12.4.1.sqsh dockerd://nccl-tests:cuda-12.4.1
   ```

### Environment File Configuration

The `.env.example` file shows configurable build parameters:

```bash
# CUDA version - controls the base CUDA image
CUDA_VERSION=12.8.1

# Component versions
GDRCOPY_VERSION=v2.5.1
EFA_INSTALLER_VERSION=1.43.2
AWS_OFI_NCCL_VERSION=v1.16.3
NCCL_VERSION=v2.27.7-1
NCCL_TESTS_VERSION=v2.16.9

# Container image name and tag
IMAGE_NAME=nccl-tests
IMAGE_TAG=cuda-${CUDA_VERSION}
```

### Using Custom CUDA Versions

To build with different CUDA versions:

```bash
# CUDA 12.4.1 (for older H200 compatibility)
docker build -f nccl-tests.Dockerfile \
  --build-arg CUDA_VERSION=12.4.1 \
  -t nccl-tests:cuda-12.4.1 .

# CUDA 12.6.2 (for newer features)
docker build -f nccl-tests.Dockerfile \
  --build-arg CUDA_VERSION=12.6.2 \
  -t nccl-tests:cuda-12.6.2 .

# CUDA 11.8.0 (for compatibility testing)
docker build -f nccl-tests.Dockerfile \
  --build-arg CUDA_VERSION=11.8.0 \
  -t nccl-tests:cuda-11.8.0 .
```

### Container Requirements

- **Docker or Podman** for building
- **Enroot** for SLURM integration
- **Shared filesystem** (e.g., /fsxl) accessible by all compute nodes
- **EFA drivers** installed on compute nodes

## Supported Operations

- **allreduce**: Combines values from all ranks and distributes result to all ranks
- **allgather**: Gathers data from all ranks and distributes to all ranks  
- **reducescatter**: Combines values and scatters results across ranks
- **alltoall**: Each rank sends different data to every other rank

## Data Patterns

- **0x0**: All zeros pattern (baseline performance, good for compression testing)
- **0x7**: Specific bit pattern (0111 binary, useful for testing data-dependent performance)

## Usage

### Basic Usage

#### Using AMI Script
```bash
# Run AllReduce with default 0x0 data pattern
sbatch nccl-tests-ami.sbatch

# Run specific operation with default 0x0 pattern
sbatch nccl-tests-ami.sbatch allreduce
sbatch nccl-tests-ami.sbatch allgather
sbatch nccl-tests-ami.sbatch reducescatter
sbatch nccl-tests-ami.sbatch alltoall
```

#### Using Container Script
```bash
# Run AllReduce with default 0x0 data pattern
sbatch nccl-tests-container.sbatch

# Run specific operation with default 0x0 pattern
sbatch nccl-tests-container.sbatch allreduce
sbatch nccl-tests-container.sbatch allgather
sbatch nccl-tests-container.sbatch reducescatter
sbatch nccl-tests-container.sbatch alltoall
```

### With Data Patterns

#### Using AMI Script
```bash
# AllReduce tests with different data patterns
sbatch nccl-tests-ami.sbatch allreduce "" 0x0
sbatch nccl-tests-ami.sbatch allreduce "" 0x7

# AllGather tests with different data patterns
sbatch nccl-tests-ami.sbatch allgather "" 0x0
sbatch nccl-tests-ami.sbatch allgather "" 0x7

# ReduceScatter tests with different data patterns
sbatch nccl-tests-ami.sbatch reducescatter "" 0x0
sbatch nccl-tests-ami.sbatch reducescatter "" 0x7

# AllToAll tests with different data patterns
sbatch nccl-tests-ami.sbatch alltoall "" 0x0
sbatch nccl-tests-ami.sbatch alltoall "" 0x7
```

#### Using Container Script
```bash
# AllReduce tests with different data patterns
sbatch nccl-tests-container.sbatch allreduce /fsxl 0x0
sbatch nccl-tests-container.sbatch allreduce /fsxl 0x7

# AllGather tests with different data patterns
sbatch nccl-tests-container.sbatch allgather /fsxl 0x0
sbatch nccl-tests-container.sbatch allgather /fsxl 0x7

# ReduceScatter tests with different data patterns
sbatch nccl-tests-container.sbatch reducescatter /fsxl 0x0
sbatch nccl-tests-container.sbatch reducescatter /fsxl 0x7

# AllToAll tests with different data patterns
sbatch nccl-tests-container.sbatch alltoall /fsxl 0x0
sbatch nccl-tests-container.sbatch alltoall /fsxl 0x7
```

## Complete Test Suite

To run all operations with both data patterns, you can submit multiple jobs:

#### AMI Test Suite
```bash
#!/bin/bash
# Run complete NCCL test suite using AMI

OPERATIONS=("allreduce" "allgather" "reducescatter" "alltoall")
PATTERNS=("0x0" "0x7")

for op in "${OPERATIONS[@]}"; do
    for pattern in "${PATTERNS[@]}"; do
        echo "Submitting ${op} test with pattern ${pattern}"
        sbatch nccl-tests-ami.sbatch ${op} "" ${pattern}
        sleep 2  # Brief delay between submissions
    done
done
```

#### Container Test Suite
```bash
#!/bin/bash
# Run complete NCCL test suite using containers

OPERATIONS=("allreduce" "allgather" "reducescatter" "alltoall")
PATTERNS=("0x0" "0x7")

for op in "${OPERATIONS[@]}"; do
    for pattern in "${PATTERNS[@]}"; do
        echo "Submitting ${op} test with pattern ${pattern}"
        sbatch nccl-tests-container.sbatch ${op} /fsxl ${pattern}
        sleep 2  # Brief delay between submissions
    done
done
```

## Parameters

Both scripts accept three parameters:

### AMI Script Parameters
1. **Test Type** (default: `allreduce`)
   - `allreduce`, `allgather`, `reducescatter`, `alltoall`

2. **Library Path** (default: `/usr/local/cuda-12.4/lib`)
   - Usually left as default (`""`)

3. **Data Pattern** (default: `0x0`)
   - `0x0` - All zeros
   - `0x7` - Bit pattern 0111
   - Other hex patterns supported

### Container Script Parameters
1. **Test Type** (default: `allreduce`)
   - `allreduce`, `allgather`, `reducescatter`, `alltoall`

2. **Apps Path** (default: `/fsxl`)
   - Path to container image location

3. **Data Pattern** (default: `0x0`)
   - `0x0` - All zeros
   - `0x7` - Bit pattern 0111
   - Other hex patterns supported

## Output

Each job will create output files:
- **AMI Script**: `nccl-tests-0x0_<job_id>.out` and `nccl-tests-0x0_<job_id>.err`
- **Container Script**: `nccl-tests-container_<job_id>.out` and `nccl-tests-container_<job_id>.err`

## Performance Results

The output will include tables showing:
- **size**: Message size in bytes
- **latency**: Communication latency in microseconds
- **busbw**: Bus bandwidth in GB/s
- **algbw**: Algorithm bandwidth in GB/s

Example output format:
```
       size         latency     busbw      algbw
          4        1193.7e-06      -         -
          8         162.7      9.8e-05      -
         16         165.1      0.000192     -
         32          246       0.000258     -
        ...
```

## Cluster Configuration

- **Nodes**: 2 x p5en.48xlarge instances
- **GPUs**: 16 x H200 GPUs total (8 per node)
- **Network**: EFA (Elastic Fabric Adapter) with RDMA
- **CUDA**: Version 12.4
- **NCCL**: Optimized with AWS OFI-NCCL plugin

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# View job output in real-time
tail -f nccl-tests-0x0_<job_id>.out

# Cancel a job if needed
scancel <job_id>
```

## Troubleshooting

1. **Job fails to start**: Check if p5en.48xlarge instances are available
2. **NCCL errors**: Verify EFA drivers and NCCL installation
3. **Performance issues**: Check network topology and GPU placement

## Performance Analysis

Compare results between 0x0 and 0x7 patterns to understand:
- Data compression effects in the network stack
- Pattern-dependent performance variations
- Bandwidth scaling across different message sizes
- Latency characteristics for small vs large messages