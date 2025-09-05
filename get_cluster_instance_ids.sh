#!/bin/bash

# Script to get AWS instance IDs for all hosts in Slurm cluster
# Run this from the Slurm head node

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Getting instance IDs for all Slurm cluster hosts...${NC}"

# Function to get instance ID from hostname using AWS CLI
get_instance_id() {
    local hostname=$1
    local instance_id
    
    # AWS CLI describe-instances by hostname
    instance_id=$(aws ec2 describe-instances \
        --filters "Name=private-dns-name,Values=$hostname" \
        --query 'Reservations[*].Instances[*].InstanceId' \
        --output text 2>/dev/null | tr -d '\t\n' || echo "")
    
    if [[ -n "$instance_id" && "$instance_id" =~ ^i-[0-9a-f]{8,17}$ ]]; then
        echo "$instance_id"
        return 0
    fi
    
    echo ""
    return 1
}

# Check if AWS CLI is available
if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: AWS CLI not found. Please install and configure AWS CLI.${NC}"
    exit 1
fi

# Check AWS credentials
if ! aws sts get-caller-identity &>/dev/null; then
    echo -e "${RED}Error: AWS credentials not configured or invalid.${NC}"
    exit 1
fi

# Get all Slurm nodes
echo -e "${YELLOW}Getting Slurm node list...${NC}"
slurm_nodes=$(sinfo -N -h -o "%N" | sort -u)

if [[ -z "$slurm_nodes" ]]; then
    echo -e "${RED}Error: No Slurm nodes found. Make sure you're on the Slurm head node.${NC}"
    exit 1
fi

echo "Found $(echo "$slurm_nodes" | wc -l) Slurm nodes"

# Create output files
output_file="cluster_instance_ids.txt"
mapping_file="hostname_to_instance_mapping.txt"
failed_file="failed_hostnames.txt"

# Clear output files
> "$output_file"
> "$mapping_file"
> "$failed_file"

echo -e "${YELLOW}Resolving instance IDs...${NC}"

# Process each node
total_nodes=$(echo "$slurm_nodes" | wc -l)
current=0
successful=0
failed=0

for hostname in $slurm_nodes; do
    current=$((current + 1))
    echo -ne "\rProcessing node $current/$total_nodes: $hostname"
    
    instance_id=$(get_instance_id "$hostname")
    
    if [[ -n "$instance_id" ]]; then
        echo "$instance_id" >> "$output_file"
        echo "$hostname $instance_id" >> "$mapping_file"
        successful=$((successful + 1))
    else
        echo "$hostname" >> "$failed_file"
        failed=$((failed + 1))
        echo -e "\n${YELLOW}Warning: Could not resolve instance ID for $hostname${NC}"
    fi
done

echo -e "\n\n${GREEN}Results:${NC}"
echo "Successfully resolved: $successful nodes"
echo "Failed to resolve: $failed nodes"

if [[ $successful -gt 0 ]]; then
    echo -e "\n${GREEN}Instance IDs saved to: $output_file${NC}"
    echo -e "${GREEN}Hostname mapping saved to: $mapping_file${NC}"
    
    echo -e "\n${YELLOW}Instance IDs:${NC}"
    cat "$output_file"
    
    echo -e "\n${YELLOW}For use in AWS CLI:${NC}"
    echo "aws ec2 describe-instance-topology --instance-ids $(tr '\n' ' ' < "$output_file")"
fi

if [[ $failed -gt 0 ]]; then
    echo -e "\n${RED}Failed hostnames saved to: $failed_file${NC}"
    echo -e "${RED}Failed hostnames:${NC}"
    cat "$failed_file"
fi

# Create a JSON format output as well
if [[ $successful -gt 0 ]]; then
    json_file="cluster_instance_mapping.json"
    echo "{" > "$json_file"
    echo "  \"cluster_instances\": [" >> "$json_file"
    
    first=true
    while IFS=' ' read -r hostname instance_id; do
        if [[ "$first" == "true" ]]; then
            first=false
        else
            echo "," >> "$json_file"
        fi
        echo -n "    {\"hostname\": \"$hostname\", \"instance_id\": \"$instance_id\"}" >> "$json_file"
    done < "$mapping_file"
    
    echo "" >> "$json_file"
    echo "  ]," >> "$json_file"
    echo "  \"total_instances\": $successful," >> "$json_file"
    echo "  \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"" >> "$json_file"
    echo "}" >> "$json_file"
    
    echo -e "${GREEN}JSON mapping saved to: $json_file${NC}"
fi

echo -e "\n${GREEN}Done!${NC}"