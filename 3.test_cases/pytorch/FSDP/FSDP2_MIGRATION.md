# FSDP to FSDP2 Migration Guide

## Overview

This guide covers migrating from PyTorch FSDP (v1) to FSDP2 (the composable API introduced in PyTorch 2.0+). FSDP2 provides a more composable, Pythonic API with better performance and flexibility.

## Migration Example

### Original FSDP v1 Usage
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

with torch.device("meta"):
    model = Transformer()

policy = ModuleWrapPolicy({TransformerBlock})
model = FSDP(model, auto_wrap_policy=policy)

def param_init_fn(module: nn.Module) -> None:
    ...

model = FSDP(model, auto_wrap_policy=policy, param_init_fn=param_init_fn)
```

### New FSDP2 Usage
```python
from torch.distributed._composable.fsdp import fully_shard

with torch.device("meta"):
    model = Transformer()

# Apply fully_shard to desired sublayers
for module in model.modules():
    if isinstance(module, TransformerBlock):
        fully_shard(module)

# Wrap root model
fully_shard(model)

# Verify model is still on meta device
for tensor in itertools.chain(model.parameters(), model.buffers()):
    assert tensor.device == torch.device("meta")

# Initialize the model after sharding
model.to_empty(device="cuda")
model.reset_parameters()
```

## Migration Steps

1. **Replace the imports**
2. **Implement your 'policy' directly** (apply `fully_shard` to the desired sublayers)
3. **Wrap your root model** with `fully_shard` instead of FSDP
4. **Get rid of `param_init_fn`** and manually call `model.reset_parameters()`
5. **Replace other FSDP1 kwargs** (see parameter mapping below)

## Parameter Migration Mapping

### Sharding Strategy
**FSDP v1 → FSDP2:**
- `FULL_SHARD` → `reshard_after_forward=True`
- `SHARD_GRAD_OP` → `reshard_after_forward=False`
- `HYBRID_SHARD` → `reshard_after_forward=True` with a 2D device mesh
- `_HYBRID_SHARD_ZERO2` → `reshard_after_forward=False` with a 2D device mesh

### CPU Offload
**FSDP v1 → FSDP2:**
- `CPUOffload.offload_params=False` → `offload_policy=None`
- `CPUOffload.offload_params=True` → `offload_policy=CPUOffloadPolicy()`

### Backward Prefetch
**FSDP v1 → FSDP2:**
- `BACKWARD_PRE` → Always used
- `BACKWARD_POST` → Not supported

### Mixed Precision
**FSDP v1 → FSDP2:**
- `buffer_dtype` is omitted because `fully_shard` does not shard buffers
- `fully_shard`'s `cast_forward_inputs` maps to both `cast_forward_inputs` and `cast_root_forward_inputs` in FSDP1
- `output_dtype` is a new config for `fully_shard`

### Device Management
**FSDP v1 → FSDP2:**
- `device_id` → Inferred from `device_mesh`'s devices

### State Synchronization
**FSDP v1 → FSDP2:**
- `sync_module_states=True/False` → Moved to DCP. User can broadcast state dicts from rank0 using `set_model_state_dict` with `broadcast_from_rank0=True`

### Forward Prefetch
**FSDP v1 → FSDP2:**
- Manual control over prefetching is possible with:
  - Manually call `fsdp_module.unshard()`
  - Use `set_modules_to_forward_prefetch` and `set_modules_to_backward_prefetch` APIs

### Other Parameters
**FSDP v1 → FSDP2:**
- `limit_all_gathers` → No longer needed, because `fully_shard` removed cpu synchronization
- `use_orig_params` → Original params are always used (no more flat parameter)
- `no_sync()` → `set_requires_gradient_sync`
- `ignored_params` and `ignored_states` → `ignored_params`

## Import Changes

### FSDP v1 Imports
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
```

### FSDP2 Imports
```python
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy
from torch.distributed.device_mesh import init_device_mesh
```

## Device Mesh Setup

FSDP2 uses device mesh for more flexible parallelism strategies:

```python
# Single dimension mesh for full sharding
mesh = init_device_mesh("cuda", (world_size,))

# Multi-dimension mesh for hybrid sharding
mesh = init_device_mesh("cuda", (dp_size, tp_size))
```

## Checkpointing Changes

### FSDP v1 Checkpointing
```python
from torch.distributed.fsdp import StateDictType

with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
    state_dict = model.state_dict()
```

### FSDP2 Checkpointing
```python
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict, StateDictOptions

# Get state dict
state_dict = get_state_dict(model, options=StateDictOptions(full_state_dict=False))

# Set state dict
set_state_dict(model, state_dict, options=StateDictOptions(broadcast_from_rank0=True))
```

## Files Updated for FSDP2

The following files have been updated to use FSDP2:

1. **src/train.py** - Main training script migrated to FSDP2
2. **src/model_utils/train_utils.py** - Updated utility functions for FSDP2
3. **src/model_utils/checkpoint.py** - Updated checkpoint save/load for FSDP2
4. **src/requirements.txt** - PyTorch 2.7.1+ (already compatible)
5. **README.md** - Updated documentation to reflect FSDP2 usage

## Benefits of FSDP2

1. **Composability**: Can combine with other distributed strategies (DDP, TP)
2. **Cleaner API**: More Pythonic, less wrapper-based approach
3. **Better Performance**: Improved memory efficiency and speed
4. **Flexibility**: Fine-grained control over sharding per module
5. **Future-proof**: Active development path for PyTorch
6. **Original Parameters**: Always uses original parameters (no more flat parameters)
7. **Simplified Prefetching**: Better control over forward/backward prefetching
8. **Device Mesh Integration**: Native support for multi-dimensional parallelism

## Complete Migration Checklist

- [ ] Update imports to use `torch.distributed._composable.fsdp`
- [ ] Replace `FSDP()` wrapper with `fully_shard()` calls
- [ ] Implement wrapping policy manually using loops
- [ ] Update checkpoint save/load to use DCP APIs
- [ ] Replace `param_init_fn` with manual `reset_parameters()`
- [ ] Update mixed precision configuration
- [ ] Set up device mesh for parallelism strategy
- [ ] Update gradient synchronization calls
- [ ] Test thoroughly with existing workloads
- [ ] Verify memory usage and performance improvements

## Common Pitfalls

1. **Meta Device Handling**: Ensure proper initialization after sharding
2. **Parameter Initialization**: Don't forget to call `reset_parameters()` after moving to device
3. **Device Mesh Setup**: Incorrect mesh dimensions can affect performance
4. **Checkpoint Compatibility**: FSDP1 and FSDP2 checkpoints are not directly compatible
5. **Import Paths**: FSDP2 APIs are in `torch.distributed._composable` (note the underscore)
