# Implementation Plan: CentOS 9 EFA AMI

## Overview

This implementation plan creates a CentOS 9 AMI with a complete EFA stack built from source for ML workloads. The implementation follows the existing Packer + Ansible patterns from `1.amazon_machine_image` and builds components in dependency order.

## Tasks

- [x] 1. Set up project structure and Packer configuration
  - [x] 1.1 Create directory structure for centtos9_machine_image
    - Create `2.ami_and_containers/centtos9_machine_image/` directory
    - Create `roles/`, `inventory/`, `inventory/group_vars/` subdirectories
    - _Requirements: 14.4_
  
  - [x] 1.2 Create Packer HCL configuration file
    - Create `packer-centos9.pkr.hcl` with amazon-ebs source
    - Configure base AMI ami-04331ec57720ee626, region us-east-1
    - Set instance_type to g5.16xlarge, ssh_username to ec2-user
    - Configure 100GB gp3 volume
    - Add ansible provisioner block
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_
  
  - [x] 1.3 Create Makefile for build commands
    - Add `ami_centos9_gpu` target for building the AMI
    - Add `validate` target for packer validate
    - _Requirements: 14.3_
  
  - [x] 1.4 Create inventory files
    - Create `inventory/hosts` with default group
    - Create `inventory/group_vars/all.yml` with version variables
    - _Requirements: 14.5_
  
  - [x] 1.5 Create main playbook
    - Create `playbook-centos9-gpu.yml` with role includes in dependency order
    - _Requirements: 14.5_

- [x] 2. Checkpoint - Verify project structure
  - Ensure packer validate passes, ask the user if questions arise.

- [x] 3. Implement base system roles
  - [x] 3.1 Create base_centos9 role
    - Create `roles/base_centos9/defaults/main.yml` with sysctl parameters
    - Create `roles/base_centos9/tasks/main.yml` to configure sysctl settings
    - _Requirements: 2.6_
  
  - [x] 3.2 Create packages_centos9 role
    - Create `roles/packages_centos9/defaults/main.yml` with package lists
    - Create `roles/packages_centos9/tasks/main.yml` to:
      - Enable EPEL repository for CentOS 9
      - Install development tools group (@development)
      - Install kernel-devel, kernel-headers
      - Install cmake, git, pkg-config, autoconf, automake, libtool
      - Install Python 3 and pip
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 4. Implement NVIDIA driver and CUDA roles
  - [x] 4.1 Create nvidia_driver_centos9 role
    - Create `roles/nvidia_driver_centos9/defaults/main.yml` with driver version
    - Create `roles/nvidia_driver_centos9/tasks/main.yml` to:
      - Add NVIDIA CUDA repository for RHEL 9
      - Install NVIDIA driver packages
    - _Requirements: 3.1_
  
  - [x] 4.2 Create nvidia_cuda_centos9 role
    - Create `roles/nvidia_cuda_centos9/defaults/main.yml` with CUDA version
    - Create `roles/nvidia_cuda_centos9/tasks/main.yml` to:
      - Install CUDA toolkit from NVIDIA repo
      - Create /etc/profile.d/cuda.sh with PATH, LD_LIBRARY_PATH, CPATH
    - _Requirements: 3.2, 3.3, 3.4_
  
  - [x] 4.3 Create nvidia_gdrcopy_centos9 role
    - Create `roles/nvidia_gdrcopy_centos9/defaults/main.yml` with GDRCopy settings
    - Create `roles/nvidia_gdrcopy_centos9/tasks/main.yml` to:
      - Install GDRCopy build dependencies
      - Clone GDRCopy from GitHub
      - Build and install to /usr/local/gdrcopy
      - Load kernel module with insmod.sh
      - Create /etc/profile.d/gdrcopy.sh
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 5. Checkpoint - Verify NVIDIA stack
  - Ensure all NVIDIA roles have correct structure, ask the user if questions arise.

- [x] 6. Implement EFA driver and rdma-core roles
  - [x] 6.1 Create efa_driver_src role
    - Create `roles/efa_driver_src/defaults/main.yml` with repo URL and branch
    - Create `roles/efa_driver_src/tasks/main.yml` to:
      - Clone amzn-drivers from https://github.com/amzn/amzn-drivers
      - Build EFA kernel module from kernel/linux/efa directory
      - Install kernel module
      - Configure module to load at boot via /etc/modules-load.d/
    - _Requirements: 5.1, 5.2, 5.3, 5.4_
  
  - [x] 6.2 Create rdma_core_src role
    - Create `roles/rdma_core_src/defaults/main.yml` with version
    - Create `roles/rdma_core_src/tasks/main.yml` to:
      - Install rdma-core build dependencies (ninja-build, python3-docutils, etc.)
      - Clone rdma-core from upstream
      - Build with cmake
      - Install libraries and headers
      - Configure /etc/ld.so.conf.d/rdma-core.conf
      - Run ldconfig
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [x] 7. Implement hwloc role
  - [x] 7.1 Create hwloc_src role
    - Create `roles/hwloc_src/defaults/main.yml` with version 2.10.0
    - Create `roles/hwloc_src/tasks/main.yml` to:
      - Download hwloc release tarball
      - Extract and configure with --prefix=/opt/hwloc
      - Build and install
      - Create /etc/profile.d/hwloc.sh
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [x] 8. Implement libfabric role
  - [x] 8.1 Create libfabric_src role
    - Create `roles/libfabric_src/defaults/main.yml` with:
      - Version v2.4.0
      - Prefix /opt/amazon/efa
      - All configure flags
      - Providers list
    - Create `roles/libfabric_src/tasks/main.yml` to:
      - Clone libfabric v2.4.0 from https://github.com/ofiwg/libfabric
      - Run autogen.sh
      - Configure with all specified flags:
        - --prefix=/opt/amazon/efa
        - --with-cuda='/usr/local/cuda'
        - --enable-cuda-dlopen
        - --with-gdrcopy=/usr/local/gdrcopy
        - --enable-gdrcopy-dlopen
        - --enable-efa
        - --disable-verbs --disable-psm3 --disable-opx --disable-usnic --disable-rstream
      - Build with PROVIDERS_TO_BUILD='verbs efa rxd shm sm2 lpp perf trace monitor hook_debug hook_hmem dmabuf_peer_mem'
      - Install and run ldconfig
      - Create /etc/profile.d/libfabric.sh
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 8.10_

- [x] 9. Checkpoint - Verify EFA stack
  - Ensure EFA driver, rdma-core, hwloc, and libfabric roles are complete, ask the user if questions arise.

- [x] 10. Implement NCCL and jemalloc roles
  - [x] 10.1 Create nccl_src role
    - Create `roles/nccl_src/defaults/main.yml` with:
      - Version v2.28.3-1 (or appropriate 2.28.x tag)
      - Prefix /opt/nccl
      - NVCC_GENCODE for compute 70, 75, 80, 90
    - Create `roles/nccl_src/tasks/main.yml` to:
      - Clone NCCL v2.28 from NVIDIA repository
      - Build with make src.build and CUDA_HOME, NVCC_GENCODE
      - Create /etc/profile.d/nccl.sh
    - _Requirements: 9.1, 9.2, 9.3, 9.4_
  
  - [x] 10.2 Create jemalloc_src role
    - Create `roles/jemalloc_src/defaults/main.yml` with version 5.3.0
    - Create `roles/jemalloc_src/tasks/main.yml` to:
      - Download jemalloc 5.3.0 source tarball
      - Configure with --enable-static --prefix=/opt/jemalloc
      - Build and install static library
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [x] 11. Implement aws-ofi-nccl role
  - [x] 11.1 Create aws_ofi_nccl_src role
    - Create `roles/aws_ofi_nccl_src/defaults/main.yml` with:
      - Repository URL
      - Prefix /opt/aws-ofi-nccl
      - Paths to libfabric, NCCL, CUDA, jemalloc
    - Create `roles/aws_ofi_nccl_src/tasks/main.yml` to:
      - Clone aws-ofi-nccl from https://github.com/aws/aws-ofi-nccl
      - Run autogen.sh
      - Configure with:
        - --prefix=/opt/aws-ofi-nccl
        - --with-libfabric=/opt/amazon/efa
        - --with-nccl=/opt/nccl/build
        - --with-cuda=/usr/local/cuda
        - LDFLAGS for static jemalloc linking
      - Build and install
      - Create /etc/profile.d/aws-ofi-nccl.sh
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7_

- [x] 12. Implement NCCL tests role
  - [x] 12.1 Create nccl_tests_src role
    - Create `roles/nccl_tests_src/defaults/main.yml` with:
      - Commit 9a5c15461abcef145b907c54d04aea4e8d1cb21f
      - Prefix /opt/nccl-tests
    - Create `roles/nccl_tests_src/tasks/main.yml` to:
      - Clone nccl-tests and checkout specific commit
      - Build with MPI=1, CUDA_HOME, MPI_HOME, NCCL_HOME
      - Configure for static NCCL linking (NCCL_LIB_DIR, STATIC_NCCL=1 or BUILDDIR with static lib)
      - Install to /opt/nccl-tests
    - _Requirements: 12.1, 12.2, 12.3, 12.4_

- [x] 13. Implement environment configuration role
  - [x] 13.1 Create environment_config role
    - Create `roles/environment_config/defaults/main.yml` with all paths
    - Create `roles/environment_config/tasks/main.yml` to:
      - Create consolidated /etc/profile.d/efa-stack.sh with all environment variables
      - Configure LD_LIBRARY_PATH with all library paths
      - Configure PATH with all binary paths
      - Configure CPATH with all header paths
      - Run ldconfig
    - Create `roles/environment_config/templates/efa-stack.sh.j2` template
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5_

- [x] 14. Final checkpoint - Verify complete implementation
  - Ensure all roles are complete and playbook includes all roles in correct order, ask the user if questions arise.

- [x] 15. Create ParallelCluster build script
  - [x] 15.1 Create pcluster-build-ami.sh script
    - Create `2.ami_and_containers/centtos9_machine_image/pcluster-build-ami.sh`
    - Accept AMI ID as required first argument
    - Accept optional --region parameter (default: us-east-1)
    - Accept optional --image-name parameter for output AMI name
    - Accept optional --instance-type parameter for build instance
    - Check for pcluster CLI availability, exit with error if not found
    - Generate temporary YAML config file with ParentImage set to input AMI
    - Execute `pcluster build-image` with generated config
    - Poll `pcluster describe-image` every 60 seconds to monitor progress
    - Display build status updates during the process
    - Output final AMI ID on successful completion
    - Clean up temporary config file
    - Provide clear error messages on failure
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5, 15.6, 15.7, 15.8, 15.9, 15.10, 15.11_

- [x] 16. Write property tests
  - [x] 16.1 Write property test for environment configuration completeness
    - **Property 1: Environment Configuration Completeness**
    - **Validates: Requirements 13.2, 13.3, 13.4**
  
  - [x] 16.2 Write property test for role structure consistency
    - **Property 2: Role Structure Consistency**
    - **Validates: Requirements 14.2, 14.5**
  
  - [x] 16.3 Write property test for source build configuration completeness
    - **Property 3: Source Build Configuration Completeness**
    - **Validates: Requirements 5.1-5.4, 6.1-6.4, 7.1-7.4, 8.1-8.10, 9.1-9.4, 10.1-10.4, 11.1-11.7, 12.1-12.4**

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- The implementation follows the existing patterns from `1.amazon_machine_image`
- All source builds use specific version tags/commits for reproducibility
- The g5.16xlarge instance type is used for the build to ensure GPU driver installation works correctly
