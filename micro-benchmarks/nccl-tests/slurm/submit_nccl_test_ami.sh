#!/bin/bash

# Script to submit comprehensive NCCL tests with AMI-based jobs
# Tests all collective operations with different data patterns

set -e

# Configuration
NODE_COUNTS=(2)
ADDITIONAL_LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib"
TEST_TYPES=("allreduce" "allgather" "reducescatter" "alltoall")
DATA_PATTERNS=("0x0" "0x7")

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting comprehensive NCCL test submission (AMI version)...${NC}"
echo -e "${BLUE}Configuration:${NC}"
echo "  Node counts: ${NODE_COUNTS[*]}"
echo "  LD Library path: $ADDITIONAL_LD_LIBRARY_PATH"
echo "  Test types: ${TEST_TYPES[*]}"
echo "  Data patterns: ${DATA_PATTERNS[*]}"
echo ""

# Counter for submitted jobs
job_count=0
submitted_jobs=()

# Create job tracking files
timestamp=$(date +"%Y%m%d_%H%M%S")
job_ids_file="logs/submitted_jobs_ami_${timestamp}.txt"
job_details_file="logs/job_details_ami_${timestamp}.csv"

# Initialize CSV file with headers
echo "JobID,Nodes,TestType,DataPattern,TotalGPUs,SubmissionTime" > "$job_details_file"

# Submit all test combinations
for nodes in "${NODE_COUNTS[@]}"; do
    total_gpus=$((nodes * 8))
    
    echo -e "${YELLOW}=== Submitting AMI tests for $nodes nodes ($total_gpus GPUs) ===${NC}"
    
    for test_type in "${TEST_TYPES[@]}"; do
        for data_pattern in "${DATA_PATTERNS[@]}"; do
            echo "Submitting: $test_type with pattern $data_pattern on $nodes nodes"
            
            # Submit the job and capture job ID
            job_output=$(sbatch --nodes=$nodes nccl-tests-ami.sbatch "$test_type" "$ADDITIONAL_LD_LIBRARY_PATH" "$data_pattern")
            job_id=$(echo "$job_output" | grep -o '[0-9]\+')
            
            if [ -n "$job_id" ]; then
                submitted_jobs+=("$job_id")
                job_count=$((job_count + 1))
                echo "  → Job ID: $job_id"
                
                # Save job ID to file
                echo "$job_id" >> "$job_ids_file"
                
                # Save job details to CSV
                submission_time=$(date +"%Y-%m-%d %H:%M:%S")
                echo "$job_id,$nodes,$test_type,$data_pattern,$total_gpus,$submission_time" >> "$job_details_file"
            else
                echo "  → Error: Failed to get job ID"
            fi
            
            # Small delay to avoid overwhelming the scheduler
            sleep 1
        done
    done
    echo ""
done

echo -e "${GREEN}Summary:${NC}"
echo "Total jobs submitted: $job_count"
echo "Job IDs: ${submitted_jobs[*]}"
echo ""

# Save summary information
echo -e "${BLUE}Job tracking files created:${NC}"
echo "  Job IDs: $job_ids_file"
echo "  Job details: $job_details_file"
echo ""

# Show queue status
echo -e "${YELLOW}Current queue status:${NC}"
squeue -u $USER

echo ""
echo -e "${GREEN}All jobs submitted successfully!${NC}"
echo -e "${BLUE}Monitor progress with: squeue -u $USER${NC}"
echo -e "${BLUE}Check job details with: scontrol show job <job_id>${NC}"
echo -e "${BLUE}Monitor specific jobs: squeue -j $(IFS=,; echo "${submitted_jobs[*]}")${NC}"
echo ""
echo -e "${YELLOW}To automatically process results as jobs complete, run:${NC}"
echo -e "${BLUE}./process_nccl_results.sh $job_ids_file${NC}"
echo ""
echo -e "${YELLOW}To cancel all submitted jobs if needed:${NC}"
echo -e "${BLUE}scancel $(IFS=' '; echo "${submitted_jobs[*]}")${NC}"