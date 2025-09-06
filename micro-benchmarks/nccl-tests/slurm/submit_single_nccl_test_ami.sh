#!/bin/bash

# Script to submit a single NCCL test job with AMI
# 2 nodes, allreduce, 0x0 data pattern

set -e

# Configuration
NODES=2
TEST_TYPE="allreduce"
DATA_PATTERN="0x0"
ADDITIONAL_LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}Submitting single NCCL test (AMI version)...${NC}"
echo -e "${BLUE}Configuration:${NC}"
echo "  Nodes: $NODES"
echo "  Test type: $TEST_TYPE"
echo "  Data pattern: $DATA_PATTERN"
echo "  LD Library path: $ADDITIONAL_LD_LIBRARY_PATH"
echo ""

total_gpus=$((NODES * 8))
echo -e "${YELLOW}Submitting $TEST_TYPE test with pattern $DATA_PATTERN on $NODES nodes ($total_gpus GPUs)${NC}"

# Submit the job and capture job ID
echo "sbatch --nodes=$NODES nccl-tests-ami.sbatch "$TEST_TYPE" "$ADDITIONAL_LD_LIBRARY_PATH" "$DATA_PATTERN""
job_output=$(sbatch --nodes=$NODES nccl-tests-ami.sbatch "$TEST_TYPE" "$ADDITIONAL_LD_LIBRARY_PATH" "$DATA_PATTERN")
job_id=$(echo "$job_output" | grep -o '[0-9]\+')

if [ -n "$job_id" ]; then
    echo -e "${GREEN}Job submitted successfully!${NC}"
    echo "  → Job ID: $job_id"
    
    # Create simple tracking file
    timestamp=$(date +"%Y%m%d_%H%M%S")
    echo "$job_id" > "logs/single_job_${timestamp}.txt"
    
    echo ""
    echo -e "${BLUE}Job tracking file created: logs/single_job_${timestamp}.txt${NC}"
    echo ""
    
    # Show queue status
    echo -e "${YELLOW}Current queue status:${NC}"
    squeue -u $USER
    
    echo ""
    echo -e "${GREEN}Job submitted successfully!${NC}"
    echo -e "${BLUE}Monitor progress with: squeue -u $USER${NC}"
    echo -e "${BLUE}Check job details with: scontrol show job $job_id${NC}"
    echo -e "${BLUE}Monitor this job: squeue -j $job_id${NC}"
    echo ""
    echo -e "${YELLOW}To cancel this job if needed:${NC}"
    echo -e "${BLUE}scancel $job_id${NC}"
else
    echo -e "${RED}Error: Failed to submit job${NC}"
    exit 1
fi