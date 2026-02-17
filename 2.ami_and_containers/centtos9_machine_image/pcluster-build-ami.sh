#!/bin/bash
#
# pcluster-build-ami.sh
#
# Script to build a ParallelCluster-ready AMI from a custom base AMI.
# Uses pcluster build-image to add ParallelCluster components to the AMI.
#
# Usage:
#   ./pcluster-build-ami.sh <ami-id> [options]
#
# Options:
#   --region <region>         AWS region (default: us-east-1)
#   --image-name <name>       Name for the output AMI
#   --instance-type <type>    Instance type for build process
#   -h, --help                Show this help message
#

set -euo pipefail

# Default values
DEFAULT_REGION="us-east-1"
POLL_INTERVAL=60

# Script variables
AMI_ID=""
REGION="${DEFAULT_REGION}"
IMAGE_NAME=""
INSTANCE_TYPE=""
CONFIG_FILE=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

#######################################
# Print usage information
#######################################
usage() {
    cat << EOF
Usage: $(basename "$0") <ami-id> [options]

Build a ParallelCluster-ready AMI from a custom base AMI.

Arguments:
  ami-id                    Required. The source AMI ID to use as parent image.

Options:
  --region <region>         AWS region (default: ${DEFAULT_REGION})
  --image-name <name>       Name for the output AMI (default: auto-generated)
  --instance-type <type>    Instance type for the build process
  -h, --help                Show this help message

Examples:
  $(basename "$0") ami-0123456789abcdef0
  $(basename "$0") ami-0123456789abcdef0 --region us-west-2
  $(basename "$0") ami-0123456789abcdef0 --image-name my-pcluster-ami --instance-type g5.xlarge
EOF
}

#######################################
# Print error message and exit
#######################################
error() {
    echo -e "${RED}ERROR: $1${NC}" >&2
    exit 1
}

#######################################
# Print warning message
#######################################
warn() {
    echo -e "${YELLOW}WARNING: $1${NC}" >&2
}

#######################################
# Print info message
#######################################
info() {
    echo -e "${GREEN}INFO: $1${NC}"
}

#######################################
# Print status message
#######################################
status() {
    echo -e "STATUS: $1"
}

#######################################
# Check if pcluster CLI is available
#######################################
check_pcluster() {
    if ! command -v pcluster &> /dev/null; then
        error "pcluster CLI is not installed or not in PATH. Please install AWS ParallelCluster CLI first."
    fi
    info "pcluster CLI found: $(command -v pcluster)"
}

#######################################
# Validate AMI ID format
#######################################
validate_ami_id() {
    local ami_id="$1"
    if [[ ! "$ami_id" =~ ^ami-[a-f0-9]{8,17}$ ]]; then
        error "Invalid AMI ID format: ${ami_id}. Expected format: ami-xxxxxxxxxxxxxxxxx"
    fi
}

#######################################
# Generate temporary config file
#######################################
generate_config() {
    CONFIG_FILE=$(mktemp /tmp/pcluster-build-config.XXXXXX.yaml)
    
    info "Generating temporary configuration file: ${CONFIG_FILE}"
    
    cat > "${CONFIG_FILE}" << EOF
Build:
  InstanceType: ${INSTANCE_TYPE:-g5.xlarge}
  ParentImage: ${AMI_ID}
EOF

    # Add optional image name if specified
    if [[ -n "${IMAGE_NAME}" ]]; then
        cat >> "${CONFIG_FILE}" << EOF
  Tags:
    - Key: Name
      Value: ${IMAGE_NAME}
EOF
    fi
    
    info "Configuration file contents:"
    cat "${CONFIG_FILE}"
}

#######################################
# Cleanup temporary files
#######################################
cleanup() {
    if [[ -n "${CONFIG_FILE}" && -f "${CONFIG_FILE}" ]]; then
        info "Cleaning up temporary config file: ${CONFIG_FILE}"
        rm -f "${CONFIG_FILE}"
    fi
}

#######################################
# Start the build process
#######################################
start_build() {
    local build_name
    
    # Generate a unique build name if image name not provided
    if [[ -n "${IMAGE_NAME}" ]]; then
        build_name="${IMAGE_NAME}"
    else
        build_name="pcluster-build-$(date +%Y%m%d-%H%M%S)"
    fi
    
    info "Starting ParallelCluster image build: ${build_name}"
    info "Parent AMI: ${AMI_ID}"
    info "Region: ${REGION}"
    
    # Execute pcluster build-image
    if ! pcluster build-image \
        --image-id "${build_name}" \
        --image-configuration "${CONFIG_FILE}" \
        --region "${REGION}"; then
        error "Failed to start pcluster build-image"
    fi
    
    echo "${build_name}"
}

#######################################
# Poll build status until completion
#######################################
poll_build_status() {
    local build_name="$1"
    local status_output
    local build_status
    local ami_id
    
    info "Monitoring build progress (polling every ${POLL_INTERVAL} seconds)..."
    
    while true; do
        # Get current status
        status_output=$(pcluster describe-image --image-id "${build_name}" --region "${REGION}" 2>&1) || {
            error "Failed to get build status for ${build_name}"
        }
        
        # Extract build status
        build_status=$(echo "${status_output}" | grep -oP '"imageBuildStatus":\s*"\K[^"]+' || echo "UNKNOWN")
        
        status "Build status: ${build_status}"
        
        case "${build_status}" in
            "BUILD_IN_PROGRESS")
                status "Build in progress... waiting ${POLL_INTERVAL} seconds"
                sleep "${POLL_INTERVAL}"
                ;;
            "BUILD_COMPLETE")
                info "Build completed successfully!"
                # Extract the AMI ID from the output
                ami_id=$(echo "${status_output}" | grep -oP '"ec2AmiInfo".*?"amiId":\s*"\K[^"]+' || echo "")
                if [[ -n "${ami_id}" ]]; then
                    echo ""
                    echo "=========================================="
                    info "ParallelCluster AMI build successful!"
                    echo "=========================================="
                    echo ""
                    echo "Output AMI ID: ${ami_id}"
                    echo ""
                    return 0
                else
                    # Try alternative parsing
                    ami_id=$(echo "${status_output}" | grep -oP '"amiId":\s*"\K[^"]+' || echo "")
                    if [[ -n "${ami_id}" ]]; then
                        echo ""
                        echo "=========================================="
                        info "ParallelCluster AMI build successful!"
                        echo "=========================================="
                        echo ""
                        echo "Output AMI ID: ${ami_id}"
                        echo ""
                        return 0
                    fi
                    warn "Build completed but could not extract AMI ID from output"
                    echo "Full output:"
                    echo "${status_output}"
                    return 0
                fi
                ;;
            "BUILD_FAILED")
                echo ""
                echo "=========================================="
                error "Build failed! Check the build logs for details."
                echo "=========================================="
                echo ""
                echo "Build details:"
                echo "${status_output}"
                return 1
                ;;
            "DELETE_IN_PROGRESS"|"DELETE_COMPLETE"|"DELETE_FAILED")
                error "Build was deleted. Status: ${build_status}"
                ;;
            *)
                warn "Unknown build status: ${build_status}"
                status "Continuing to poll..."
                sleep "${POLL_INTERVAL}"
                ;;
        esac
    done
}

#######################################
# Parse command line arguments
#######################################
parse_args() {
    if [[ $# -eq 0 ]]; then
        usage
        exit 1
    fi
    
    # First argument should be AMI ID (unless it's help)
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            error "First argument must be an AMI ID, not an option. Use -h for help."
            ;;
        *)
            AMI_ID="$1"
            shift
            ;;
    esac
    
    # Parse remaining options
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --region)
                if [[ -z "${2:-}" ]]; then
                    error "--region requires a value"
                fi
                REGION="$2"
                shift 2
                ;;
            --image-name)
                if [[ -z "${2:-}" ]]; then
                    error "--image-name requires a value"
                fi
                IMAGE_NAME="$2"
                shift 2
                ;;
            --instance-type)
                if [[ -z "${2:-}" ]]; then
                    error "--instance-type requires a value"
                fi
                INSTANCE_TYPE="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                error "Unknown option: $1. Use -h for help."
                ;;
        esac
    done
}

#######################################
# Main function
#######################################
main() {
    # Parse command line arguments
    parse_args "$@"
    
    # Validate AMI ID
    validate_ami_id "${AMI_ID}"
    
    # Check for pcluster CLI
    check_pcluster
    
    # Set up cleanup trap
    trap cleanup EXIT
    
    # Generate config file
    generate_config
    
    # Start the build
    local build_name
    build_name=$(start_build)
    
    # Poll for completion
    poll_build_status "${build_name}"
}

# Run main function
main "$@"
