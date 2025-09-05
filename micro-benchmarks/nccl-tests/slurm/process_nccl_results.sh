#!/bin/bash

# Script to monitor specific NCCL test jobs and automatically convert outputs to Excel
# Usage: ./process_nccl_results.sh <submitted_jobs_file.txt>
# Example: ./process_nccl_results.sh submitted_jobs_20250905_052718.txt

set -e

# Check arguments
if [ $# -ne 1 ]; then
    echo "Usage: $0 <submitted_jobs_file.txt>"
    echo "Example: $0 submitted_jobs_20250905_052718.txt"
    echo ""
    echo "Available job files:"
    ls -1 submitted_jobs_*.txt 2>/dev/null || echo "  No submitted_jobs_*.txt files found"
    exit 1
fi

JOBS_FILE="$1"

# Validate input file
if [[ ! -f "$JOBS_FILE" ]]; then
    echo "Error: Job file '$JOBS_FILE' not found"
    exit 1
fi

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
RESULTS_DIR="nccl_results"
CSV_CONVERTER="../nccl_to_csv.py"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Read job IDs from file
mapfile -t JOB_IDS < "$JOBS_FILE"

echo -e "${GREEN}NCCL Results Processor${NC}"
echo -e "${BLUE}Processing jobs from: $JOBS_FILE${NC}"
echo -e "${BLUE}Job IDs to monitor: ${JOB_IDS[*]}${NC}"
echo ""

# Function to extract job parameters from output file content only
parse_job_details() {
    local output_file=$1
    local nodes test_type data_pattern
    
    if [[ ! -f "$output_file" ]]; then
        echo "unknown_unknown_unknown"
        return
    fi
    
    # Extract test type from output
    if grep -q "Running NCCL.*test" "$output_file"; then
        test_type=$(grep "Running NCCL.*test" "$output_file" | sed -n 's/.*Running NCCL \([a-z]*\) test.*/\1/p' | head -1)
    fi
    
    # Extract data pattern from output
    if grep -q "data pattern" "$output_file"; then
        data_pattern=$(grep "data pattern" "$output_file" | sed -n 's/.*data pattern \(0x[0-9a-fA-F]*\).*/\1/p' | head -1)
    fi
    
    # Count unique hostnames to determine nodes
    if grep -q "p5en-dy-gpu-" "$output_file"; then
        nodes=$(grep -o "p5en-dy-gpu-[0-9]*" "$output_file" | sort -u | wc -l)
    fi
    
    echo "${nodes:-unknown}_${test_type:-unknown}_${data_pattern:-unknown}"
}

# Function to convert output to CSV
convert_to_csv() {
    local output_file=$1
    local job_details=$2
    
    echo -e "${YELLOW}Converting $output_file to CSV...${NC}"
    
    # Check if converter exists
    if [[ ! -f "$CSV_CONVERTER" ]]; then
        echo -e "${RED}Error: CSV converter not found at $CSV_CONVERTER${NC}"
        return 1
    fi
    
    # Run converter
    if python3 "$CSV_CONVERTER" "$output_file"; then
        # Move generated files to results directory with descriptive names
        local base_name=$(basename "$output_file" .out)
        
        if [[ -f "${base_name}_results.csv" ]]; then
            mv "${base_name}_results.csv" "$RESULTS_DIR/nccl_${job_details}_${TIMESTAMP}_results.csv"
            echo -e "${GREEN}  → Results: $RESULTS_DIR/nccl_${job_details}_${TIMESTAMP}_results.csv${NC}"
        fi
        
        if [[ -f "${base_name}_results_summary.csv" ]]; then
            mv "${base_name}_results_summary.csv" "$RESULTS_DIR/nccl_${job_details}_${TIMESTAMP}_summary.csv"
            echo -e "${GREEN}  → Summary: $RESULTS_DIR/nccl_${job_details}_${TIMESTAMP}_summary.csv${NC}"
        fi
        
        return 0
    else
        echo -e "${RED}  → Conversion failed${NC}"
        return 1
    fi
}

# Function to check if output file has performance data
has_performance_data() {
    local output_file=$1
    
    if [[ ! -f "$output_file" ]]; then
        return 1
    fi
    
    # Check for NCCL performance table
    if grep -q "out-of-place.*in-place" "$output_file" && \
       grep -q "size.*count.*type.*redop" "$output_file" && \
       grep -q "Avg bus bandwidth" "$output_file"; then
        return 0
    fi
    
    return 1
}

# Removed job status checking - assuming all jobs are complete

# Function to get expected output filename for job ID
get_output_filename() {
    local job_id=$1
    echo "nccl-tests-container_${job_id}.out"
}

# Main monitoring loop
processed_files=()
completed_jobs=()
failed_jobs=()

echo -e "${BLUE}Processing ${#JOB_IDS[@]} completed jobs...${NC}"
echo -e "${BLUE}Timestamp for this run: ${TIMESTAMP}${NC}"
echo ""

# Process all jobs assuming they are complete
for job_id in "${JOB_IDS[@]}"; do
    output_file=$(get_output_filename "$job_id")
    
    echo -e "${YELLOW}Processing job $job_id...${NC}"
    
    if [[ -f "$output_file" ]] && has_performance_data "$output_file"; then
        job_details=$(parse_job_details "$output_file")
        echo -e "${BLUE}  → Job details: $job_details${NC}"
        
        if convert_to_csv "$output_file" "$job_details"; then
            processed_files+=("$output_file")
            completed_jobs+=("$job_id")
            echo -e "${GREEN}  → Successfully processed job $job_id${NC}"
        else
            failed_jobs+=("$job_id")
            echo -e "${RED}  → Processing failed for job $job_id${NC}"
        fi
    else
        echo -e "${YELLOW}  → Output file missing or incomplete for job $job_id${NC}"
        failed_jobs+=("$job_id")
    fi
    echo ""
done
while true; do
    running_count=0
    pending_count=0
    
    for job_id in "${JOB_IDS[@]}"; do
        # Skip if already processed
        if [[ " ${completed_jobs[*]} " =~ " ${job_id} " ]] || [[ " ${failed_jobs[*]} " =~ " ${job_id} " ]]; then
            continue
        fi
        
        status=$(check_job_status "$job_id")
        output_file=$(get_output_filename "$job_id")
        
        case "$status" in
            "RUNNING")
                running_count=$((running_count + 1))
                ;;
            "PENDING"|"CONFIGURING")
                pending_count=$((pending_count + 1))
                ;;
            "COMPLETED")
                echo -e "${GREEN}Job $job_id completed. Processing output...${NC}"
                
                if [[ -f "$output_file" ]] && has_performance_data "$output_file"; then
                    job_details=$(parse_job_details "$output_file")
                    echo -e "${BLUE}  → Job details: $job_details${NC}"
                    
                    if convert_to_csv "$output_file" "$job_details"; then
                        processed_files+=("$output_file")
                        completed_jobs+=("$job_id")
                        echo -e "${GREEN}  → Successfully processed job $job_id${NC}"
                    else
                        failed_jobs+=("$job_id")
                        echo -e "${RED}  → Processing failed for job $job_id${NC}"
                    fi
                else
                    echo -e "${YELLOW}  → Output file missing or incomplete for job $job_id${NC}"
                    failed_jobs+=("$job_id")
                fi
                ;;
            "FAILED"|"CANCELLED"|"TIMEOUT"|"NODE_FAIL")
                echo -e "${RED}Job $job_id failed with status: $status${NC}"
                failed_jobs+=("$job_id")
                ;;
            "NOT_FOUND")
                echo -e "${YELLOW}Job $job_id not found (may have completed and been purged)${NC}"
                # Check if output file exists and process it
                if [[ -f "$output_file" ]] && has_performance_data "$output_file"; then
                    job_details=$(parse_job_details "$output_file")
                    if convert_to_csv "$output_file" "$job_details"; then
                        processed_files+=("$output_file")
                        completed_jobs+=("$job_id")
                        echo -e "${GREEN}  → Found and processed output for job $job_id${NC}"
                    fi
                else
                    failed_jobs+=("$job_id")
                fi
                ;;
        esac
    done
    
    # Check if all jobs are done
    total_done=$((${#completed_jobs[@]} + ${#failed_jobs[@]}))
    if [[ $total_done -eq ${#JOB_IDS[@]} ]]; then
        echo -e "${GREEN}All jobs processed!${NC}"
        break
    fi
    
    echo -e "${BLUE}Status: $running_count running, $pending_count pending, ${#completed_jobs[@]} completed, ${#failed_jobs[@]} failed${NC}"
    echo -e "${BLUE}Checking again in 30 seconds...${NC}"
    sleep 30
done

echo ""
echo -e "${GREEN}Processing complete!${NC}"
echo -e "${BLUE}Results saved in: $RESULTS_DIR/${NC}"

echo ""
echo -e "${GREEN}Summary:${NC}"
echo "  Successfully processed: ${#completed_jobs[@]} jobs"
echo "  Failed/Missing: ${#failed_jobs[@]} jobs"
echo "  Total jobs: ${#JOB_IDS[@]}"

if [[ ${#completed_jobs[@]} -gt 0 ]]; then
    echo ""
    echo -e "${GREEN}Successfully processed jobs:${NC}"
    for job_id in "${completed_jobs[@]}"; do
        echo "  - Job $job_id"
    done
fi

if [[ ${#failed_jobs[@]} -gt 0 ]]; then
    echo ""
    echo -e "${RED}Failed/Missing jobs:${NC}"
    for job_id in "${failed_jobs[@]}"; do
        echo "  - Job $job_id"
    done
fi

if [[ ${#processed_files[@]} -gt 0 ]]; then
    echo ""
    echo -e "${BLUE}Generated Excel files:${NC}"
    ls -la "$RESULTS_DIR"/*.xls 2>/dev/null || echo "No Excel files found"
else
    echo -e "${YELLOW}No Excel files were generated${NC}"
fi