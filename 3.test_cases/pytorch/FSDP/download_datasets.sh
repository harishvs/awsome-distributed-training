#!/bin/bash

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# Standalone script to pre-download datasets before training
# This avoids rate limiting issues during distributed training when happens when 
# you use a streaming dataset on HF
#
# Usage: ./download_datasets.sh <model_name>
# Example: ./download_datasets.sh llama3_2_1b

set -e

echo "=========================================="
echo "Dataset Pre-download Script"
echo "=========================================="
echo ""

# Check if model name is provided
if [ -z "$1" ]; then
    echo "ERROR: Model name is required"
    echo "Usage: $0 <model_name>"
    echo ""
    echo "Available models:"
    ls models/*.txt 2>/dev/null | sed 's|models/||g' | sed 's|.txt||g' | sed 's/^/  - /'
    exit 1
fi

MODEL_NAME="$1"

# Find the models directory
if [ -d "models" ]; then
    MODELS_DIR="models"
else
    echo "ERROR: Cannot find models directory"
    echo "Please run this script from the FSDP directory"
    exit 1
fi

MODEL_CONFIG_FILE="${MODELS_DIR}/${MODEL_NAME}.txt"

# Check if model config file exists
if [ ! -f "$MODEL_CONFIG_FILE" ]; then
    echo "ERROR: Model configuration file not found: $MODEL_CONFIG_FILE"
    echo ""
    echo "Available models:"
    ls ${MODELS_DIR}/*.txt 2>/dev/null | sed "s|${MODELS_DIR}/||g" | sed 's|.txt||g' | sed 's/^/  - /'
    exit 1
fi

echo "Using model configuration: $MODEL_NAME"
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

# Extract configuration from model file
DATASET=$(grep "^--dataset=" "$MODEL_CONFIG_FILE" | cut -d'=' -f2)
DATASET_CONFIG=$(grep "^--dataset_config_name=" "$MODEL_CONFIG_FILE" | cut -d'=' -f2)
TOKENIZER=$(grep "^--tokenizer=" "$MODEL_CONFIG_FILE" | cut -d'=' -f2)

# Fallback to defaults if not found in model config
DATASET=${DATASET:-"allenai/c4"}
DATASET_CONFIG=${DATASET_CONFIG:-"en"}
TOKENIZER=${TOKENIZER:-"hf-internal-testing/llama-tokenizer"}
SPLITS=${SPLITS:-"train validation"}
DATA_PATH=${DATA_PATH:-"/fsx"}

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Config: $DATASET_CONFIG"
echo "  Tokenizer: $TOKENIZER"
echo "  Splits: $SPLITS"
echo "  Data Path: $DATA_PATH"
echo "  Cache: $DATA_PATH/.cache/huggingface/datasets"
echo ""

# Check if running in container
if [ ! -z "$CONTAINER_IMAGE" ]; then
    echo "Running in container mode..."
    python src/download_dataset.py \
        --dataset "$DATASET" \
        --dataset_config_name "$DATASET_CONFIG" \
        --tokenizer "$TOKENIZER" \
        --splits $SPLITS \
        --cache_dir "$DATA_PATH/.cache/huggingface/datasets"
else
    # Check if we're in the right directory
    if [ -f "src/download_dataset.py" ]; then
        python src/download_dataset.py \
            --dataset "$DATASET" \
            --dataset_config_name "$DATASET_CONFIG" \
            --tokenizer "$TOKENIZER" \
            --splits $SPLITS \
            --cache_dir "$DATA_PATH/.cache/huggingface/datasets"
    elif [ -f "../src/download_dataset.py" ]; then
        python ../src/download_dataset.py \
            --dataset "$DATASET" \
            --dataset_config_name "$DATASET_CONFIG" \
            --tokenizer "$TOKENIZER" \
            --splits $SPLITS \
            --cache_dir "$DATA_PATH/.cache/huggingface/datasets"
    else
        echo "ERROR: Cannot find download_dataset.py"
        echo "Please run this script from the FSDP directory"
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "Dataset download complete for $MODEL_NAME!"
echo "You can now run your training job."
echo "=========================================="
