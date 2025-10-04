#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Standalone script to pre-download datasets before training
# This avoids rate limiting issues during distributed training

set -e

echo "=========================================="
echo "Dataset Pre-download Script"
echo "=========================================="
echo ""

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set. You may encounter rate limits."
    echo "To set it: export HF_TOKEN='hf_your_token_here'"
    echo ""
else
    echo "✓ HF_TOKEN is set"
    echo ""
fi

# Default dataset configuration
DATASET=${DATASET:-"allenai/c4"}
DATASET_CONFIG=${DATASET_CONFIG:-"en"}
TOKENIZER=${TOKENIZER:-"hf-internal-testing/llama-tokenizer"}
SPLITS=${SPLITS:-"train validation"}

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Config: $DATASET_CONFIG"
echo "  Tokenizer: $TOKENIZER"
echo "  Splits: $SPLITS"
echo "  Cache: /fsxl/.cache/huggingface/datasets"
echo ""

# Check if running in container
if [ ! -z "$CONTAINER_IMAGE" ]; then
    echo "Running in container mode..."
    python ../src/download_dataset.py \
        --dataset "$DATASET" \
        --dataset_config_name "$DATASET_CONFIG" \
        --tokenizer "$TOKENIZER" \
        --splits $SPLITS \
        --cache_dir /fsxl/.cache/huggingface/datasets
else
    # Check if we're in the right directory
    if [ -f "../src/download_dataset.py" ]; then
        python ../src/download_dataset.py \
            --dataset "$DATASET" \
            --dataset_config_name "$DATASET_CONFIG" \
            --tokenizer "$TOKENIZER" \
            --splits $SPLITS \
            --cache_dir /fsxl/.cache/huggingface/datasets
    elif [ -f "src/download_dataset.py" ]; then
        python src/download_dataset.py \
            --dataset "$DATASET" \
            --dataset_config_name "$DATASET_CONFIG" \
            --tokenizer "$TOKENIZER" \
            --splits $SPLITS \
            --cache_dir /fsxl/.cache/huggingface/datasets
    else
        echo "ERROR: Cannot find download_dataset.py"
        echo "Please run this script from the FSDP directory or slurm subdirectory"
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "Dataset download complete!"
echo "You can now run your training job."
echo "=========================================="
