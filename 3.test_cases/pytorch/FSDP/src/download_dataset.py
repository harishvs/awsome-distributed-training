#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Pre-download datasets to avoid rate limiting during distributed training.
This script should be run once before launching the training job.
"""

import argparse
import os
import sys
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Download HuggingFace datasets and tokenizers for training")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., allenai/c4)"
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="Dataset configuration name (e.g., en for C4)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer model name (e.g., meta-llama/Llama-3.2-1B)"
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "validation"],
        help="Dataset splits to download (default: train validation)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/fsxl/.cache/huggingface/datasets",
        help="Cache directory for datasets (default: /fsxl/.cache/huggingface/datasets) - use /fsxl when running on host"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Get HuggingFace token from environment if available
    hf_token = os.environ.get("HF_TOKEN", None)
    if hf_token:
        print("Using HF_TOKEN from environment")
    else:
        print("No HF_TOKEN found - using anonymous access (may have lower rate limits)")
    
    # Set cache directory for transformers
    tokenizer_cache_dir = args.cache_dir.replace("/datasets", "")
    os.environ['HF_HOME'] = tokenizer_cache_dir
    os.environ['TRANSFORMERS_CACHE'] = tokenizer_cache_dir
    
    # Download tokenizer if specified
    if args.tokenizer:
        print(f"\nDownloading tokenizer: {args.tokenizer}")
        print(f"Tokenizer cache directory: {tokenizer_cache_dir}")
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer,
                token=hf_token
            )
            print(f"✓ Tokenizer downloaded successfully")
            print()
        except Exception as e:
            print(f"✗ Failed to download tokenizer: {e}", file=sys.stderr)
            sys.exit(1)
    
    print(f"\nDownloading dataset: {args.dataset}")
    if args.dataset_config_name:
        print(f"Configuration: {args.dataset_config_name}")
    else:
        print("WARNING: No dataset_config_name specified. Some datasets require this.")
    print(f"Splits: {args.splits}")
    print(f"Cache directory: {args.cache_dir}")
    print()
    
    for split in args.splits:
        try:
            print(f"Downloading '{split}' split...")
            dataset = load_dataset(
                args.dataset,
                name=args.dataset_config_name,
                split=split,
                token=hf_token,
                cache_dir=args.cache_dir
            )
            print(f"✓ '{split}' split downloaded successfully: {len(dataset)} examples")
            print()
        except Exception as e:
            error_msg = str(e)
            print(f"✗ Failed to download '{split}' split: {error_msg}", file=sys.stderr)
            
            # Provide helpful hint if config is missing
            if "Config name is missing" in error_msg or "Please pick one among" in error_msg:
                print("\nHINT: This dataset requires a configuration name.", file=sys.stderr)
                print("Add --dataset_config_name argument, e.g.:", file=sys.stderr)
                print(f"  python {sys.argv[0]} --dataset {args.dataset} --dataset_config_name en", file=sys.stderr)
            
            sys.exit(1)
    
    print("All downloads completed successfully!")
    print(f"Cache location: {args.cache_dir}")


if __name__ == "__main__":
    main()
