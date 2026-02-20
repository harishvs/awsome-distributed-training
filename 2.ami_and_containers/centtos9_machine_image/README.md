# CentOS 9 EFA AMI Builder

Build a CentOS 9 AMI with a complete EFA (Elastic Fabric Adapter) stack compiled from source for ML workloads.

## Overview

This project creates a custom AMI optimized for distributed GPU training on AWS with:

- **NVIDIA Driver** (580.82.07) and **CUDA Toolkit** (12.4)
- **GDRCopy** for GPU Direct RDMA
- **EFA Driver** and **rdma-core** built from source
- **Libfabric** (v2.4.0) with EFA and CUDA support
- **NCCL** (v2.28.3-1) with static linking
- **AWS OFI NCCL** plugin with static jemalloc
- **NCCL Tests** for validation

### Why Build From Source?

AWS provides pre-built EFA packages for Amazon Linux and Ubuntu, but CentOS Stream 9 is not officially supported. Building from source gives us:

- Full control over component versions and compile-time options
- Optimized builds with GPU-aware networking (CUDA + GDRCopy support in libfabric)
- Static linking of jemalloc into the NCCL OFI plugin for better memory allocation performance
- Compatibility with CentOS Stream 9 kernels that don't match RHEL 9 binary packages

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Your Local Machine                        │
│                                                               │
│  Packer + Ansible ──SSH──▶  EC2 g5.16xlarge (CentOS 9)      │
│                              │                                │
│                              ├─ Phase 1: Base system          │
│                              ├─ Phase 2: NVIDIA stack         │
│                              ├─ Phase 3: EFA + RDMA           │
│                              ├─ Phase 4: hwloc + libfabric    │
│                              ├─ Phase 5: NCCL stack           │
│                              ├─ Phase 6: Tests + env          │
│                              └─ Phase 7: PCluster compat      │
│                              │                                │
│                              ▼                                │
│                         Snapshot → Base AMI                    │
│                              │                                │
│  pcluster build-image ──────▶  EC2 Image Builder             │
│                              │  (adds Slurm, scheduler)       │
│                              ▼                                │
│                         ParallelCluster AMI                   │
│                              │                                │
│  pcluster create-cluster ───▶  Head Node + Compute Fleet     │
└──────────────────────────────────────────────────────────────┘
```

### The Two-Stage Build Process

**Stage 1: Packer AMI Build (~30 min)** — Packer launches a GPU EC2 instance from a vanilla CentOS Stream 9 AMI, runs Ansible to install everything, then snapshots it. This produces the base AMI with the full EFA stack. Each component is an isolated Ansible role with its own defaults, tasks, and verification steps.

**Stage 2: ParallelCluster Image Build (~30-45 min)** — `pcluster build-image` takes the base AMI and layers ParallelCluster components on top (Slurm, auto-scaling daemon, CloudWatch agent, FSx Lustre client). The result is a ParallelCluster AMI ready for `pcluster create-cluster`.

### Component Dependency Chain

```
kernel headers
  └─▶ EFA driver (kernel module for EFA NICs)
  └─▶ GDRCopy (kernel module for GPU Direct RDMA)

NVIDIA driver
  └─▶ CUDA toolkit
       └─▶ NCCL (GPU collective comms, needs nvcc)
       └─▶ libfabric (--with-cuda for GPU memory registration)
       └─▶ GDRCopy (GPU-side DMA)

rdma-core (libibverbs, librdmacm)
  └─▶ libfabric (verbs provider for RDMA)
       └─▶ AWS OFI NCCL (bridges NCCL ↔ libfabric ↔ EFA)

hwloc (hardware topology)
  └─▶ libfabric (topology-aware provider selection)
  └─▶ AWS OFI NCCL (--with-hwloc)

jemalloc (memory allocator)
  └─▶ AWS OFI NCCL (statically linked for better alloc performance)

NCCL + MPI (OpenMPI)
  └─▶ NCCL tests (all_reduce_perf, etc.)
```

### Data Flow During Distributed Training

```
GPU 0 (Node A)                          GPU 0 (Node B)
    │                                       ▲
    ▼                                       │
  NCCL (collective op, e.g. AllReduce)    NCCL
    │                                       ▲
    ▼                                       │
  AWS OFI NCCL plugin                     AWS OFI NCCL plugin
    │                                       ▲
    ▼                                       │
  libfabric (EFA provider)               libfabric
    │                                       ▲
    ▼                                       │
  EFA driver (kernel)  ──── network ────  EFA driver
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Static jemalloc in AWS OFI NCCL | No runtime dependency, always uses optimized allocator |
| Static NCCL in NCCL tests | Self-contained binaries, no `LD_LIBRARY_PATH` needed |
| Source builds for EFA/NCCL stack | CentOS 9 kernel compatibility, custom compile flags |
| DNF packages for NVIDIA driver/CUDA | Official NVIDIA repo, well-tested, no need to customize |

## Prerequisites

- [Packer](https://www.packer.io/) >= 1.8.0
- [Ansible](https://www.ansible.com/) >= 2.14
- [AWS ParallelCluster CLI](https://docs.aws.amazon.com/parallelcluster/latest/ug/install-v3-parallelcluster.html) >= 3.13
- AWS credentials configured
- Access to launch g5.16xlarge instances

### Infrastructure Setup

Before creating a cluster, deploy the VPC and networking prerequisites. From the repo root:

```bash
# Deploy the VPC, subnets, security groups, and FSx filesystems
aws cloudformation create-stack \
  --stack-name parallelcluster-prerequisites-centos9-cluster \
  --template-body file://1.architectures/2.aws-parallelcluster/infra-templates/parallelcluster-prerequisites.yaml \
  --region us-east-1 \
  --capabilities CAPABILITY_IAM

# Wait for stack to complete
aws cloudformation wait stack-create-complete \
  --stack-name parallelcluster-prerequisites-centos9-cluster \
  --region us-east-1
```

This creates:
- VPC with public and private subnets
- Security groups for compute and FSx
- FSx for Lustre and FSx for OpenZFS filesystems

The `create-cluster.sh` script automatically looks up these resources by stack name.

## CentOS 9 Passwordless Sudo

The CentOS Stream 9 AMI from the CentOS project does not have passwordless sudo configured by default. This build uses cloud-init (`cloud-init-sudo.yaml`) to configure passwordless sudo for `ec2-user` during instance initialization:

```yaml
#cloud-config
write_files:
  - path: /etc/sudoers.d/90-cloud-init-users
    content: |
      ec2-user ALL=(ALL) NOPASSWD:ALL
    permissions: '0440'
```

This is referenced in `packer-centos9.pkr.hcl` via `user_data_file` and runs before Packer connects via SSH, enabling Ansible to use `become: true` without password prompts.

## Quick Start

```bash
# Initialize Packer plugins
make init

# Validate configuration
make validate

# Build the AMI
make ami_centos9_gpu
```

## Build Options

Override defaults with Packer variables:

```bash
packer build \
  -var "aws_region=us-west-2" \
  -var "instance_type=g5.8xlarge" \
  -var "ami_name=my-custom-ami" \
  packer-centos9.pkr.hcl
```

| Variable | Default | Description |
|----------|---------|-------------|
| `base_ami` | ami-04331ec57720ee626 | CentOS 9 base AMI |
| `aws_region` | us-east-1 | AWS region |
| `instance_type` | g5.16xlarge | Build instance type |
| `volume_size` | 100 | Root volume size (GB) |

## ParallelCluster Integration

CentOS Stream 9 is not directly supported by ParallelCluster, but it is binary-compatible with RHEL 9. Three compatibility layers bridge this gap:

1. **OS Identification** — The `pcluster_compat` role patches `/etc/os-release` to set `ID="rhel"` so ParallelCluster's OS detection passes. Original backed up to `/etc/os-release.centos9.bak`.
2. **RHUI Repository Alias** — ParallelCluster's Chef cookbook expects `codeready-builder-for-rhel-9-rhui-rpms`. CentOS uses `crb` instead. A repo alias maps the RHEL name to CentOS mirrors.
3. **FSx Lustre Client** — Pre-built `kmod-lustre-client` RPMs require RHEL 9 kernel symbols. A dummy RPM satisfies the dependency check during the ParallelCluster build.

Your custom AMI (with the full EFA stack) is used directly as the `ParentImage` — ParallelCluster adds only Slurm and its components, without duplicating EFA/NCCL.

```bash
# Build ParallelCluster AMI from your custom base
./pcluster-build-ami.sh ami-0641f2b86c7c89c31 --region us-east-1

# With custom name
./pcluster-build-ami.sh ami-0641f2b86c7c89c31 \
  --region us-east-1 \
  --image-name my-pcluster-ami
```

The original `/etc/os-release` is backed up to `/etc/os-release.centos9.bak` on the AMI.

Use the resulting AMI in your cluster configuration:
```yaml
Image:
  CustomAmi: ami-XXXXXXXXXXXXXXXXX
```

## Creating a Cluster

Once you have a ParallelCluster-ready AMI, create a cluster:

```bash
# Basic — auto-discovers VPC/subnets from parallelcluster-prerequisites-centos9-cluster stack
./create-cluster.sh ami-XXXXXXXXXXXXXXXXX

# With options
./create-cluster.sh ami-XXXXXXXXXXXXXXXXX \
  --cluster-name my-ml-cluster \
  --region us-east-1 \
  --instance-type p5.48xlarge \
  --num-instances 4 \
  --key-pair my-keypair

# Monitor cluster creation
pcluster describe-cluster --cluster-name my-ml-cluster --region us-east-1

# SSH into head node
pcluster ssh --cluster-name my-ml-cluster --region us-east-1
```

## Project Structure

```
centtos9_machine_image/
├── Makefile                    # Build commands
├── packer-centos9.pkr.hcl      # Packer configuration
├── playbook-centos9-gpu.yml    # Main Ansible playbook
├── pcluster-build-ami.sh       # ParallelCluster build script
├── create-cluster.sh           # Cluster creation script
├── inventory/
│   ├── hosts                   # Ansible inventory
│   └── group_vars/all.yml      # Version pinning
├── roles/
│   ├── base_centos9/           # System sysctl tuning
│   ├── packages_centos9/       # Development tools
│   ├── nvidia_driver_centos9/  # NVIDIA driver
│   ├── nvidia_cuda_centos9/    # CUDA toolkit
│   ├── nvidia_gdrcopy_centos9/ # GDRCopy
│   ├── efa_driver_src/         # EFA kernel module
│   ├── rdma_core_src/          # rdma-core libraries
│   ├── hwloc_src/              # Hardware topology
│   ├── libfabric_src/          # Libfabric with EFA
│   ├── nccl_src/               # NCCL
│   ├── jemalloc_src/           # Static jemalloc
│   ├── aws_ofi_nccl_src/       # AWS OFI NCCL plugin
│   ├── nccl_tests_src/         # NCCL tests
│   ├── environment_config/     # Environment setup
│   ├── lustre_client/         # FSx Lustre client compat
│   └── pcluster_compat/       # ParallelCluster OS compat
└── tests/
    └── test_properties.py      # Property-based tests
```

## Component Versions

| Component | Version | Install Path |
|-----------|---------|--------------|
| NVIDIA Driver | 580.82.07 | System |
| CUDA Toolkit | 12.4 | /usr/local/cuda |
| GDRCopy | v2.5.1 | /usr/local/gdrcopy |
| rdma-core | v52.0 | /usr |
| hwloc | 2.10.0 | /opt/hwloc |
| Libfabric | v2.4.0 | /opt/amazon/efa |
| NCCL | v2.28.3-1 | /opt/nccl |
| jemalloc | 5.3.0 | /opt/jemalloc |
| AWS OFI NCCL | master | /opt/aws-ofi-nccl |
| NCCL Tests | 9a5c154 | /opt/nccl-tests |
| OpenMPI | System | /usr/lib64/openmpi |

## Environment Variables

The AMI configures `/etc/profile.d/efa-stack.sh` with:

```bash
export CUDA_HOME=/usr/local/cuda
export FI_PROVIDER=efa
export NCCL_PROTO=simple
# PATH, LD_LIBRARY_PATH, CPATH configured for all components
```

## Testing

Run property-based tests:

```bash
cd tests
pip install pytest hypothesis
pytest test_properties.py -v
```

## Validation

After launching an instance from the AMI:

```bash
# Check EFA
fi_info -p efa

# Check NCCL
all_reduce_perf -b 8 -e 128M -f 2 -g 1
```

## Known Issues

### OpenMPI Headers on CentOS 9

CentOS 9's `openmpi-devel` package installs MPI headers to `/usr/include/openmpi-x86_64/` rather than under `/usr/lib64/openmpi/include/`. The NCCL tests build step sets `CPATH=/usr/include/openmpi-x86_64` to ensure `mpi.h` is found during compilation.

## License

See repository root for license information.
