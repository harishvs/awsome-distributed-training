# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import datetime
import functools
import itertools
import math
import os
import re
import time

import numpy as np
import torch
from torch import optim
import torch.distributed as dist
import torch.utils.data

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import DataLoader

from model_utils.concat_dataset import ConcatTokensDataset
from model_utils.train_utils import (get_model_config, 
                                   compute_num_params,
                                   get_transformer_layer,
                                   apply_activation_checkpoint,
                                   get_param_groups_by_weight_decay,
                                   get_logger,
                                   get_learning_rate_scheduler,
                                   create_dataloader)
from model_utils.checkpoint import save_checkpoint, load_checkpoint
from model_utils.arguments import parse_args


import logging
import sys

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", level=logging.INFO, stream=sys.stdout)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def eval_model(model, dataloader, num_batches):
    """Eval step."""
    model = model.eval()
    n_batches = 0
    loss = 0.0

    with torch.no_grad():
        for batch_idx, input_data in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            loss += model(input_ids=input_data, attention_mask=None, labels=input_data)["loss"]
            n_batches += 1

    if n_batches > 0:
        detached_loss = loss.detach()
        torch.distributed.all_reduce(detached_loss)
        loss = detached_loss.item() / dist.get_world_size()
        loss /= n_batches
        ppl = math.exp(loss)
    else:
        loss = -1.0
        ppl = -1.0

    return loss, ppl

def train(
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
        model_config,
        num_params,
        args,
        global_rank,
        world_size,
        total_steps=0,
        start_batch_index=0
    ):
    model.train()
    for index in range(args.epochs):
        for batch_idx, input_data in enumerate(train_dataloader):
            if batch_idx < start_batch_index:
                continue
            optimizer.zero_grad(set_to_none=True)
            step_start = time.time()
            loss = model(input_ids=input_data, attention_mask=None, labels=input_data)["loss"]
            loss.backward()
            # FSDP2 uses torch.nn.utils.clip_grad_norm_ directly
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            lr_scheduler.step()
            total_steps += 1
            loss_metric = loss.item()
            step_time = time.time() - step_start
            sample_processed = input_data.shape[0] * world_size
            throughput = sample_processed / step_time
            loss_scalar = loss.item()
            current_lr = lr_scheduler.get_lr()
            if global_rank==0 and batch_idx%args.logging_freq==0:
                logger.info(
                    "Batch %d Loss: %.5f, Speed: %.2f samples/sec, lr: %.6f",  # pylint: disable=line-too-long
                    batch_idx,
                    loss_scalar,
                    throughput,
                    current_lr,
                )
            if args.validation_freq and not total_steps % args.validation_freq:
                val_loss, val_ppl = eval_model(
                    model, val_dataloader, args.validation_batches
                )
                model = model.train()
                if global_rank == 0:
                    logger.info(
                            "Batch %d Validation loss: %s",
                            batch_idx,
                            val_loss,
                        )
            if args.checkpoint_dir and not total_steps % args.checkpoint_freq:
                user_content = {
                    "cli_args": args.__dict__,
                    "num_params": num_params,
                    "total_steps": total_steps,
                    "model_config": model_config,
                    "start_batch_index": batch_idx + 1,
                }
                sub_dir = f"{args.model_type}-{total_steps}steps"

                save_checkpoint(
                    model,
                    optimizer,
                    lr_scheduler,
                    user_content,
                    args.checkpoint_dir,
                    sub_dir,
                )
            if total_steps >= args.max_steps:
                break
            

def main(args):
    dist.init_process_group()
    global_rank = dist.get_rank()
    device = global_rank % torch.cuda.device_count()
    world_size = dist.get_world_size()
    
    if args.bf16:
        dtype = torch.bfloat16
    else:
        dtype = torch.get_default_dtype()
    
    model_config = get_model_config(args)
    if global_rank == 0:
        logger.info(
            "Creating Model"
        )
    # Instantiate model on CPU on rank=0 only to prevent CPU OOM
    # (e.g. 70B * 4 bytes * 8 processes > 2T RAM available on P5)
    if global_rank == 0:
        model = AutoModelForCausalLM.from_config(model_config)
    else:
        with torch.device("meta"):
            # Instantiating model on `meta` device doesn't consume CPU memory,
            # but requires specifing `param_init_fn=...`
            # and `sync_module_states=True` in FSDP c-tor.
            model = AutoModelForCausalLM.from_config(model_config)
    
    num_params = compute_num_params(model)
    if global_rank == 0:
        logger.info(
            "Created model with total parameters: %d (%.2f B)", num_params, num_params * 1e-9
        )
    transformer_layer = get_transformer_layer(args.model_type)

    torch.cuda.set_device(device)
    
    # Setup device mesh for FSDP2
    if args.sharding_strategy == "hybrid":
        # For hybrid sharding, use 2D mesh (assuming 8 GPUs per node)
        dp_size = world_size // 8 if world_size >= 8 else 1
        tp_size = world_size // dp_size
        mesh = init_device_mesh("cuda", (dp_size, tp_size))
        reshard_after_forward = True
    else:
        # For full sharding, use 1D mesh
        mesh = init_device_mesh("cuda", (world_size,))
        reshard_after_forward = args.sharding_strategy == "full"

    # Setup mixed precision policy for FSDP2
    mp_policy = MixedPrecisionPolicy(
        param_dtype=dtype, 
        reduce_dtype=dtype,
        cast_forward_inputs=True,
        output_dtype=dtype
    )
    
    # Setup CPU offload policy for FSDP2
    if args.cpu_offload == 1:
        offload_policy = CPUOffloadPolicy()
    else: 
        offload_policy = None

    # Apply fully_shard to transformer layers
    for module in model.modules():
        if isinstance(module, transformer_layer):
            fully_shard(
                module,
                mesh=mesh,
                mp_policy=mp_policy,
                offload_policy=offload_policy,
                reshard_after_forward=reshard_after_forward
            )

    # Apply fully_shard to root model
    fully_shard(
        model,
        mesh=mesh,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        reshard_after_forward=reshard_after_forward
    )

    # Verify model is still on meta device (only for non-rank-0 processes)
    # Rank 0 initializes on CPU to avoid OOM, while other ranks use meta device
    if global_rank != 0:
        import itertools
        for tensor in itertools.chain(model.parameters(), model.buffers()):
            assert tensor.device == torch.device("meta"), \
                f"Expected tensor on meta device, but got {tensor.device}"

    # Initialize the model after sharding
    model.to_empty(device="cuda")
    model.apply(model._init_weights)

    if global_rank == 0:
        logger.info("Wrapped model with FSDP2")

    if args.activation_checkpointing > 0:
        apply_activation_checkpoint(args, model=model)

    if args.offload_activations > 0:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import offload_wrapper

        model = offload_wrapper(model)

    param_groups = get_param_groups_by_weight_decay(model)

    optimizer = optim.AdamW(
        param_groups, betas=(args.beta1, args.beta2), lr=args.lr, weight_decay=args.weight_decay
    )

    if global_rank == 0:
        logger.info("Created optimizer")

    lr_scheduler = get_learning_rate_scheduler(optimizer, args)

    if args.resume_from_checkpoint:
        (
            model,
            optimizer,
            lr_scheduler,
            total_steps,
            start_batch_index,
        ) = load_checkpoint(model, 
                            optimizer, 
                            lr_scheduler, 
                            args.resume_from_checkpoint, 
                            args.model_type,
                            device)
    else:
        total_steps = 0
        start_batch_index = 0
    
    # Get cache directory from environment variable
    data_path = os.environ.get("DATA_PATH", "/fsx")
    cache_dir = f"{data_path}/.cache/huggingface/datasets"
    
    train_dataloader = create_dataloader(args.dataset, 
                                         args.tokenizer, 
                                         name=args.dataset_config_name, 
                                         global_rank=global_rank,
                                         batch_size=args.train_batch_size, 
                                         split='train',
                                         cache_dir=cache_dir)
    
    val_dataloader = create_dataloader(args.dataset, 
                                       args.tokenizer, 
                                       name=args.dataset_config_name,
                                       global_rank=global_rank, 
                                       batch_size=args.train_batch_size, 
                                       split='validation',
                                       cache_dir=cache_dir)
    
    train(model, 
          optimizer, 
          train_dataloader,
          val_dataloader,
          lr_scheduler, 
          model_config, 
          num_params, 
          args, 
          global_rank, 
          world_size,
          total_steps,
          start_batch_index)
  
    dist.destroy_process_group()

if __name__ == "__main__":
    args, _ = parse_args()
    main(args)
