#!/bin/bash

# NCCL-based Bad Node Finder
# This script uses NCCL all-reduce tests to identify problematic nodes

set -e

LOGDIR="find_bad_nodes_logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="$LOGDIR/analysis_summary_$TIMESTAMP.txt"
COMBINATIONS_FILE="$LOGDIR/node_combinations_$TIMESTAMP.txt"

mkdir -p "$LOGDIR"

echo "NCCL Bad Node Finder - Started at $(date)" | tee "$SUMMARY_FILE"

# Configuration
MIN_NODES=${MIN_NODES:-2}
MAX_NODES=${MAX_NODES:-8}
TEST_DURATION=${TEST_DURATION:-30}  # seconds
NCCL_TEST_PATH=${NCCL_TEST_PATH:-"/opt/nccl-tests/build/all_reduce_perf"}

# Get available nodes
if command -v sinfo &>/dev/null; then
    AVAILABLE_NODES=($(sinfo -N -h -t idle,alloc -o "%N" | sort -u))
else
    echo "Error: Slurm not available. Please set AVAILABLE_NODES manually."
    exit 1
fi

echo "Found ${#AVAILABLE_NODES[@]} available nodes: ${AVAILABLE_NODES[*]}" | tee -a "$SUMMARY_FILE"

# Function to run NCCL test on specific nodes
run_nccl_test() {
    local nodes=("$@")
    local node_list=$(IFS=,; echo "${nodes[*]}")
    local num_nodes=${#nodes[@]}
    local gpus_per_node=8  # Adjust based on your instance type
    local total_gpus=$((num_nodes * gpus_per_node))
    
    echo "Testing nodes: $node_list" | tee -a "$COMBINATIONS_FILE"
    
    # Create temporary hostfile
    local hostfile=$(mktemp)
    for node in "${nodes[@]}"; do
        for ((i=0; i<gpus_per_node; i++)); do
            echo "$node slots=1" >> "$hostfile"
        done
    done
    
    # Run NCCL test
    local output_file="$LOGDIR/nccl_test_${num_nodes}nodes_$TIMESTAMP.log"
    
    timeout $TEST_DURATION mpirun \
        --hostfile "$hostfile" \
        --map-by ppr:$gpus_per_node:node \
        --bind-to none \
        -x NCCL_DEBUG=INFO \
        -x NCCL_TREE_THRESHOLD=0 \
        -x NCCL_IB_DISABLE=1 \
        -x NCCL_SOCKET_IFNAME=^docker0,lo \
        "$NCCL_TEST_PATH" \
        -b 1G -e 1G -f 2 -g 1 \
        > "$output_file" 2>&1
    
    local exit_code=$?
    rm -f "$hostfile"
    
    # Analyze results
    if [[ $exit_code -eq 0 ]]; then
        local bandwidth=$(grep "Avg bus bandwidth" "$output_file" | tail -1 | awk '{print $6}')
        echo "  ✅ SUCCESS - Bandwidth: ${bandwidth:-N/A} GB/s" | tee -a "$COMBINATIONS_FILE"
        return 0
    else
        echo "  ❌ FAILED - Exit code: $exit_code" | tee -a "$COMBINATIONS_FILE"
        
        # Check for specific error patterns
        if grep -q "NCCL WARN" "$output_file"; then
            echo "    NCCL warnings detected" | tee -a "$COMBINATIONS_FILE"
        fi
        if grep -q "timeout" "$output_file"; then
            echo "    Timeout detected" | tee -a "$COMBINATIONS_FILE"
        fi
        
        return 1
    fi
}

# Function to find bad nodes using binary search approach
find_bad_nodes() {
    local all_nodes=("$@")
    local bad_nodes=()
    
    echo "" | tee -a "$SUMMARY_FILE"
    echo "=== BINARY SEARCH FOR BAD NODES ===" | tee -a "$SUMMARY_FILE"
    
    # Test all nodes together first
    echo "Testing all ${#all_nodes[@]} nodes together..." | tee -a "$SUMMARY_FILE"
    if run_nccl_test "${all_nodes[@]}"; then
        echo "✅ All nodes working together" | tee -a "$SUMMARY_FILE"
        return 0
    fi
    
    echo "❌ Issues detected with full node set. Starting binary search..." | tee -a "$SUMMARY_FILE"
    
    # Binary search to isolate bad nodes
    local nodes_to_test=("${all_nodes[@]}")
    
    while [[ ${#nodes_to_test[@]} -gt 1 ]]; do
        local mid=$((${#nodes_to_test[@]} / 2))
        local first_half=("${nodes_to_test[@]:0:$mid}")
        local second_half=("${nodes_to_test[@]:$mid}")
        
        echo "Testing first half: ${first_half[*]}" | tee -a "$SUMMARY_FILE"
        if ! run_nccl_test "${first_half[@]}"; then
            nodes_to_test=("${first_half[@]}")
            echo "  Issue in first half" | tee -a "$SUMMARY_FILE"
        else
            echo "Testing second half: ${second_half[*]}" | tee -a "$SUMMARY_FILE"
            if ! run_nccl_test "${second_half[@]}"; then
                nodes_to_test=("${second_half[@]}")
                echo "  Issue in second half" | tee -a "$SUMMARY_FILE"
            else
                echo "  Both halves work individually - may be a scaling issue" | tee -a "$SUMMARY_FILE"
                break
            fi
        fi
    done
    
    # Test individual nodes if we narrowed it down
    if [[ ${#nodes_to_test[@]} -eq 1 ]]; then
        echo "Testing individual node: ${nodes_to_test[0]}" | tee -a "$SUMMARY_FILE"
        if ! run_nccl_test "${nodes_to_test[0]}"; then
            bad_nodes+=("${nodes_to_test[0]}")
        fi
    fi
    
    # Report bad nodes
    if [[ ${#bad_nodes[@]} -gt 0 ]]; then
        echo "" | tee -a "$SUMMARY_FILE"
        echo "❌ BAD NODES IDENTIFIED: ${bad_nodes[*]}" | tee -a "$SUMMARY_FILE"
        echo "Recommended action: scontrol update NodeName=${bad_nodes[*]} State=drain Reason='NCCL test failed'"
        return 1
    else
        echo "✅ No individual bad nodes found - may be a configuration or scaling issue" | tee -a "$SUMMARY_FILE"
        return 0
    fi
}

# Main execution
echo "Starting NCCL-based bad node detection..." | tee -a "$SUMMARY_FILE"

# Limit to reasonable number of nodes for testing
if [[ ${#AVAILABLE_NODES[@]} -gt $MAX_NODES ]]; then
    TEST_NODES=("${AVAILABLE_NODES[@]:0:$MAX_NODES}")
    echo "Limiting test to first $MAX_NODES nodes: ${TEST_NODES[*]}" | tee -a "$SUMMARY_FILE"
else
    TEST_NODES=("${AVAILABLE_NODES[@]}")
fi

find_bad_nodes "${TEST_NODES[@]}"

echo "" | tee -a "$SUMMARY_FILE"
echo "Analysis complete at $(date)" | tee -a "$SUMMARY_FILE"
echo "Logs saved in: $LOGDIR"
echo "Summary: $SUMMARY_FILE"