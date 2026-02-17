# Requirements Document

## Introduction

This document specifies the requirements for building a CentOS 9 AMI with a complete EFA (Elastic Fabric Adapter) stack built entirely from source for ML workloads. The AMI will include the EFA driver, rdma-core, libfabric, NCCL, aws-ofi-nccl plugin, and NCCL tests, all compiled from source with specific configurations optimized for GPU-accelerated distributed training.

## Glossary

- **AMI_Builder**: The Packer-based system that creates Amazon Machine Images
- **Ansible_Provisioner**: The Ansible playbook and roles system that configures the AMI
- **EFA_Driver**: The Elastic Fabric Adapter kernel driver from amzn-drivers repository
- **RDMA_Core**: The userspace RDMA library providing libibverbs and librdmacm
- **Libfabric**: The Open Fabrics Interfaces library providing fabric communication abstraction
- **NCCL**: NVIDIA Collective Communications Library for multi-GPU/multi-node communication
- **AWS_OFI_NCCL**: AWS OFI NCCL plugin that enables NCCL to use libfabric for network communication
- **GDRCopy**: NVIDIA GPU Direct RDMA copy library for low-latency GPU memory transfers
- **Hwloc**: Hardware Locality library for portable hardware topology abstraction
- **Jemalloc**: Memory allocator designed for scalability and fragmentation avoidance
- **NCCL_Tests**: NVIDIA NCCL test suite for validating collective operations

## Requirements

### Requirement 1: AMI Base Configuration

**User Story:** As a DevOps engineer, I want to build an AMI from a specific CentOS 9 base image, so that I have a consistent and reproducible foundation for ML workloads.

#### Acceptance Criteria

1. THE AMI_Builder SHALL use base AMI ami-04331ec57720ee626 in us-east-1 region
2. THE AMI_Builder SHALL use Packer with the Amazon EBS builder plugin
3. THE AMI_Builder SHALL use Ansible as the provisioner for configuration management
4. THE AMI_Builder SHALL configure SSH access using the appropriate username for CentOS 9
5. THE AMI_Builder SHALL provision adequate disk space (minimum 100GB) for build artifacts and installed software

### Requirement 2: CentOS 9 Base System Packages

**User Story:** As a system administrator, I want the base system configured with all necessary development tools and dependencies, so that all source builds can complete successfully.

#### Acceptance Criteria

1. THE Ansible_Provisioner SHALL install development tools group (gcc, make, autoconf, automake, libtool)
2. THE Ansible_Provisioner SHALL install kernel development headers matching the running kernel
3. THE Ansible_Provisioner SHALL install required build dependencies (cmake, git, pkg-config)
4. THE Ansible_Provisioner SHALL configure EPEL repository for additional packages
5. THE Ansible_Provisioner SHALL install Python 3 and pip for build scripts
6. THE Ansible_Provisioner SHALL configure system sysctl parameters for optimal network performance

### Requirement 3: NVIDIA Driver and CUDA Installation

**User Story:** As an ML engineer, I want NVIDIA drivers and CUDA toolkit installed, so that GPU acceleration is available for ML workloads.

#### Acceptance Criteria

1. THE Ansible_Provisioner SHALL install NVIDIA GPU drivers compatible with CentOS 9
2. THE Ansible_Provisioner SHALL install CUDA toolkit from NVIDIA repositories
3. THE Ansible_Provisioner SHALL configure CUDA environment variables in /etc/profile.d/
4. THE Ansible_Provisioner SHALL verify CUDA installation path at /usr/local/cuda

### Requirement 4: GDRCopy Installation

**User Story:** As an ML engineer, I want GDRCopy installed, so that GPU Direct RDMA operations can achieve low-latency memory transfers.

#### Acceptance Criteria

1. THE Ansible_Provisioner SHALL build GDRCopy from source
2. THE Ansible_Provisioner SHALL install GDRCopy to /usr/local/gdrcopy prefix
3. THE Ansible_Provisioner SHALL configure GDRCopy environment variables in /etc/profile.d/
4. THE Ansible_Provisioner SHALL load GDRCopy kernel module

### Requirement 5: EFA Driver from Source

**User Story:** As a system administrator, I want the EFA driver built from source, so that I have full control over the driver version and build configuration.

#### Acceptance Criteria

1. THE Ansible_Provisioner SHALL clone the EFA driver from https://github.com/amzn/amzn-drivers
2. THE Ansible_Provisioner SHALL build the EFA kernel module from source
3. THE Ansible_Provisioner SHALL install the EFA kernel module
4. THE Ansible_Provisioner SHALL configure the EFA module to load at boot
5. WHEN the EFA driver build fails, THEN the Ansible_Provisioner SHALL report a clear error message

### Requirement 6: rdma-core from Source

**User Story:** As a system administrator, I want rdma-core built from upstream source, so that I have the latest RDMA userspace libraries.

#### Acceptance Criteria

1. THE Ansible_Provisioner SHALL clone rdma-core from upstream repository
2. THE Ansible_Provisioner SHALL build rdma-core with cmake
3. THE Ansible_Provisioner SHALL install rdma-core libraries and headers
4. THE Ansible_Provisioner SHALL configure rdma-core library paths in /etc/ld.so.conf.d/

### Requirement 7: Hwloc from Source

**User Story:** As an ML engineer, I want hwloc installed, so that applications can query hardware topology for optimal process placement.

#### Acceptance Criteria

1. THE Ansible_Provisioner SHALL download hwloc release version source
2. THE Ansible_Provisioner SHALL build hwloc with standard configuration
3. THE Ansible_Provisioner SHALL install hwloc to /opt/hwloc prefix
4. THE Ansible_Provisioner SHALL configure hwloc environment variables in /etc/profile.d/

### Requirement 8: Libfabric 2.4.0 from Source

**User Story:** As an ML engineer, I want libfabric built with specific EFA and CUDA options, so that the fabric layer is optimized for GPU-accelerated distributed training.

#### Acceptance Criteria

1. THE Ansible_Provisioner SHALL clone libfabric version 2.4.0 from https://github.com/ofiwg/libfabric
2. THE Ansible_Provisioner SHALL configure libfabric with prefix /opt/amazon/efa
3. THE Ansible_Provisioner SHALL configure libfabric with --with-cuda='/usr/local/cuda'
4. THE Ansible_Provisioner SHALL configure libfabric with --enable-cuda-dlopen
5. THE Ansible_Provisioner SHALL configure libfabric with --with-gdrcopy pointing to GDRCopy installation
6. THE Ansible_Provisioner SHALL configure libfabric with --enable-gdrcopy-dlopen
7. THE Ansible_Provisioner SHALL configure libfabric with --enable-efa
8. THE Ansible_Provisioner SHALL configure libfabric with --disable-verbs --disable-psm3 --disable-opx --disable-usnic --disable-rstream
9. THE Ansible_Provisioner SHALL build libfabric with PROVIDERS_TO_BUILD='verbs efa rxd shm sm2 lpp perf trace monitor hook_debug hook_hmem dmabuf_peer_mem'
10. THE Ansible_Provisioner SHALL install libfabric and configure library paths
11. WHEN libfabric build fails due to missing dependencies, THEN the Ansible_Provisioner SHALL report which dependency is missing

### Requirement 9: NCCL 2.28 from Source

**User Story:** As an ML engineer, I want NCCL 2.28 built from source, so that I have the specific version required for my distributed training workloads.

#### Acceptance Criteria

1. THE Ansible_Provisioner SHALL clone NCCL version 2.28 from NVIDIA repository
2. THE Ansible_Provisioner SHALL build NCCL with CUDA support for compute capabilities 70, 75, 80, 90
3. THE Ansible_Provisioner SHALL install NCCL to /opt/nccl prefix
4. THE Ansible_Provisioner SHALL configure NCCL environment variables in /etc/profile.d/
5. WHEN NCCL build fails, THEN the Ansible_Provisioner SHALL report the build error

### Requirement 10: Jemalloc 5.3.0 Static Build

**User Story:** As an ML engineer, I want jemalloc built as a static library, so that it can be statically linked into the aws-ofi-nccl plugin.

#### Acceptance Criteria

1. THE Ansible_Provisioner SHALL download jemalloc version 5.3.0 source
2. THE Ansible_Provisioner SHALL build jemalloc as a static library
3. THE Ansible_Provisioner SHALL install jemalloc static library and headers
4. THE Ansible_Provisioner SHALL make jemalloc available for static linking by aws-ofi-nccl

### Requirement 11: AWS OFI NCCL Plugin with Static Jemalloc

**User Story:** As an ML engineer, I want the aws-ofi-nccl plugin built with statically linked jemalloc, so that NCCL can use EFA for network communication with optimized memory allocation.

#### Acceptance Criteria

1. THE Ansible_Provisioner SHALL clone aws-ofi-nccl from https://github.com/aws/aws-ofi-nccl
2. THE Ansible_Provisioner SHALL configure aws-ofi-nccl with libfabric from /opt/amazon/efa
3. THE Ansible_Provisioner SHALL configure aws-ofi-nccl with NCCL from /opt/nccl
4. THE Ansible_Provisioner SHALL configure aws-ofi-nccl with CUDA from /usr/local/cuda
5. THE Ansible_Provisioner SHALL statically link jemalloc 5.3.0 into the aws-ofi-nccl plugin
6. THE Ansible_Provisioner SHALL install aws-ofi-nccl to /opt/aws-ofi-nccl prefix
7. THE Ansible_Provisioner SHALL configure aws-ofi-nccl library paths in /etc/profile.d/

### Requirement 12: NCCL Tests with Static NCCL

**User Story:** As an ML engineer, I want NCCL tests built with statically linked NCCL, so that I can validate collective operations without runtime library dependencies.

#### Acceptance Criteria

1. THE Ansible_Provisioner SHALL clone NCCL tests at commit 9a5c15461abcef145b907c54d04aea4e8d1cb21f
2. THE Ansible_Provisioner SHALL build NCCL tests with MPI support
3. THE Ansible_Provisioner SHALL statically link NCCL into the test binaries
4. THE Ansible_Provisioner SHALL install NCCL tests to /opt/nccl-tests
5. WHEN NCCL tests build fails, THEN the Ansible_Provisioner SHALL report the build error

### Requirement 13: Environment Configuration

**User Story:** As a user, I want all environment variables properly configured, so that all installed components are accessible without manual configuration.

#### Acceptance Criteria

1. THE Ansible_Provisioner SHALL create /etc/profile.d/ scripts for all installed components
2. THE Ansible_Provisioner SHALL configure LD_LIBRARY_PATH to include all library paths
3. THE Ansible_Provisioner SHALL configure PATH to include all binary paths
4. THE Ansible_Provisioner SHALL configure CPATH to include all header paths
5. THE Ansible_Provisioner SHALL run ldconfig to update library cache
6. WHEN a user logs in, THEN all environment variables SHALL be automatically set

### Requirement 14: Project Structure Consistency

**User Story:** As a DevOps engineer, I want the project structure to follow existing patterns, so that the codebase remains consistent and maintainable.

#### Acceptance Criteria

1. THE AMI_Builder SHALL use Packer HCL format for AMI definition
2. THE Ansible_Provisioner SHALL organize roles with defaults/, tasks/, and files/ subdirectories
3. THE AMI_Builder SHALL provide a Makefile for build commands
4. THE AMI_Builder SHALL place all files in 2.ami_and_containers/centtos9_machine_image directory
5. THE Ansible_Provisioner SHALL use role-based organization consistent with 1.amazon_machine_image structure

### Requirement 15: ParallelCluster Build Script

**User Story:** As a cluster administrator, I want a script that takes an AMI ID and makes it ParallelCluster-ready using `pcluster build-image`, so that I can easily deploy the custom AMI in ParallelCluster environments.

#### Acceptance Criteria

1. THE script SHALL accept an AMI ID as a required input parameter
2. THE script SHALL accept an optional AWS region parameter (default: us-east-1)
3. THE script SHALL accept an optional image name parameter for the output AMI
4. THE script SHALL accept an optional instance type parameter for the build process
5. THE script SHALL generate a temporary YAML configuration file for pcluster build-image
6. THE script SHALL execute `pcluster build-image` with the generated configuration
7. THE script SHALL poll `pcluster describe-image` to monitor build progress
8. THE script SHALL display the final AMI ID upon successful completion
9. THE script SHALL provide clear error messages if the build fails
10. THE script SHALL clean up temporary files after completion
11. WHEN the pcluster CLI is not installed, THEN the script SHALL exit with an error message
