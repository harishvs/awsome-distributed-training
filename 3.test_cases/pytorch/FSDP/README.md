# Get Started Training Llama 2, Mixtral 8x7B, and Mistral Mathstral with PyTorch FSDP2 in 5 Minutes

This content provides a quickstart with multinode PyTorch [FSDP2](https://pytorch.org/docs/stable/fsdp.html) (composable API) training on Slurm and Kubernetes.
It is designed to be simple with no data preparation or tokenizer to download, and uses Python virtual environment.

**Note**: This implementation has been migrated from FSDP v1 to FSDP2 (the composable API). See [FSDP2_MIGRATION.md](FSDP2_MIGRATION.md) for migration details.

## Prerequisites

To run FSDP2 training, you will need to create a training cluster based on Slurm or Kubermetes with an [Amazon FSx for Lustre](https://docs.aws.amazon.com/fsx/latest/LustreGuide/what-is.html)
You can find instruction how to create a Amazon SageMaker Hyperpod cluster with [Slurm](https://catalog.workshops.aws/sagemaker-hyperpod/en-US), [Kubernetes](https://catalog.workshops.aws/sagemaker-hyperpod-eks/en-US) or with in [Amazon EKS](../../1.architectures).

## FSDP2 Training

This folder provides examples on how to train with PyTorch FSDP2 (composable API) with Slurm or Kubernetes.
You will find instructions for [Slurm](slurm) or [Kubernetes](kubernetes) in the subdirectories.

### Key FSDP2 Features Used

- **Composable API**: Uses `fully_shard()` instead of FSDP wrapper
- **Device Mesh**: Supports both 1D (full sharding) and 2D (hybrid sharding) parallelism
- **Original Parameters**: Always uses original parameters (no flat parameters)
- **Improved Checkpointing**: Uses distributed checkpoint (DCP) APIs

## Migration from FSDP v1

This implementation has been updated to use FSDP2 (composable API). Key changes include:

- Replaced `FullyShardedDataParallel` wrapper with `fully_shard()` function calls
- Updated imports to use `torch.distributed._composable.fsdp`
- Migrated to device mesh for parallelism configuration
- Updated checkpointing to use distributed checkpoint (DCP) APIs
- Simplified parameter initialization with `reset_parameters()`

For detailed migration information, see [FSDP2_MIGRATION.md](FSDP2_MIGRATION.md).