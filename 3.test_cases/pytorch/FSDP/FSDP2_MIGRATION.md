# FSDP to FSDP2 Migration Guide

## Overview

This guide covers migrating from PyTorch FSDP (v1) to FSDP2 (the composable API introduced in PyTorch 2.0+).

## Key Differences

### 1. Import Changes
**FSDP v1:**
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
```

**FSDP2:**
```python
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh
```

### 2. Model Wrapping
**FSDP v1:** Wraps the entire model as a single object
```python
model = FSDP(
    model,
    auto_wrap_policy=gpt_auto_wrap_policy,
    mixed_precision=mixed_precision_policy,
    sharding_strategy=sharding_strategy,
    device_id=torch.cuda.current_device(),
)
```

**FSDP2:** Applies sharding to individual modules in-place
```python
for module in model.modules():
    if isinstance(module, TransformerLayer):
        fully_shard(module, mesh=mesh, mp_policy=mp_policy)
fully_shard(model, mesh=mesh, mp_policy=mp_policy)
```

### 3. Device Mesh
FSDP2 uses device mesh for more flexible parallelism strategies:
```python
mesh = init_device_mesh("cuda", (world_size,))
```

### 4. Mixed Precision
**FSDP v1:**
```python
mixed_precision_policy = MixedPrecision(
    param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype
)
```

**FSDP2:**
```python
mp_policy = MixedPrecisionPolicy(
    param_dtype=dtype, reduce_dtype=dtype
)
```

### 5. Checkpointing
**FSDP v1:** Uses `StateDictType` context manager
```python
with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
    state_dict = model.state_dict()
```

**FSDP2:** Uses `state_dict_type` parameter
```python
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict, StateDictOptions
state_dict = get_state_dict(model, options=StateDictOptions(full_state_dict=False))
```

### 6. Sharding Strategies
**FSDP v1:**
- `ShardingStrategy.FULL_SHARD`
- `ShardingStrategy.HYBRID_SHARD`

**FSDP2:**
- Controlled via device mesh dimensions
- Single dimension mesh = full sharding
- Multi-dimension mesh = hybrid sharding

### 7. CPU Offload
**FSDP v1:**
```python
cpu_offload = CPUOffload(offload_params=True)
```

**FSDP2:**
```python
# Pass offload_policy parameter to fully_shard
from torch.distributed._composable.fsdp import OffloadPolicy
offload_policy = OffloadPolicy(offload_params=True)
```

## Files Requiring Changes. create new files , dont change existing ones

1. **src/train.py** - Main training script
2. **src/model_utils/train_utils.py** - Utility functions
3. **src/model_utils/checkpoint.py** - Checkpoint save/load
4. **src/requirements.txt** - Ensure PyTorch 2.0+

## Benefits of FSDP2

1. **Composability**: Can combine with other distributed strategies (DDP, TP)
2. **Cleaner API**: More Pythonic, less wrapper-based
3. **Better Performance**: Improved memory efficiency and speed
4. **Flexibility**: Fine-grained control over sharding per module
5. **Future-proof**: Active development path for PyTorch

## Migration Steps

1. Update imports
2. Replace FSDP wrapper with fully_shard calls
3. Update checkpoint save/load logic
4. Update mixed precision configuration
5. Test thoroughly with existing workloads
