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

## Prerequisites

- [Packer](https://www.packer.io/) >= 1.8.0
- [Ansible](https://www.ansible.com/) >= 2.14
- AWS credentials configured
- Access to launch g5.16xlarge instances

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

After building the base AMI, make it ParallelCluster-ready:

```bash
./pcluster-build-ami.sh ami-0123456789abcdef0 \
  --region us-east-1 \
  --image-name my-pcluster-ami
```

## Project Structure

```
centtos9_machine_image/
├── Makefile                    # Build commands
├── packer-centos9.pkr.hcl      # Packer configuration
├── playbook-centos9-gpu.yml    # Main Ansible playbook
├── pcluster-build-ami.sh       # ParallelCluster build script
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
│   └── environment_config/     # Environment setup
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

## License

See repository root for license information.
