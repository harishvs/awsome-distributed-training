#!/bin/bash
#
# pcluster-build-ami.sh
#
# Build a ParallelCluster-ready AMI from a custom CentOS 9 / RHEL 9 base AMI.
#
# The base AMI must have os-release patched to identify as RHEL 9
# (handled by the pcluster_compat Ansible role). This allows ParallelCluster
# to accept it directly as a ParentImage without duplicating the EFA stack.
#
# Usage:
#   ./pcluster-build-ami.sh <ami-id> [options]
#

set -euo pipefail

DEFAULT_REGION="us-east-1"
POLL_INTERVAL=60

AMI_ID=""
REGION="${DEFAULT_REGION}"
IMAGE_NAME=""
INSTANCE_TYPE=""
CONFIG_FILE=""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

usage() {
    cat << EOF
Usage: $(basename "$0") <ami-id> [options]

Build a ParallelCluster-ready AMI from a custom base AMI.

The base AMI should have /etc/os-release patched to identify as RHEL 9
(done automatically by the pcluster_compat Ansible role during the Packer build).

Arguments:
  ami-id                    Required. The custom AMI ID with EFA stack.

Options:
  --region <region>         AWS region (default: ${DEFAULT_REGION})
  --image-name <name>       Name for the output AMI (default: auto-generated)
  --instance-type <type>    Instance type for the build process (default: g5.xlarge)
  -h, --help                Show this help message

Examples:
  $(basename "$0") ami-0123456789abcdef0
  $(basename "$0") ami-0123456789abcdef0 --region us-west-2
  $(basename "$0") ami-0123456789abcdef0 --image-name my-pcluster-ami
EOF
}

error() { echo -e "${RED}ERROR: $1${NC}" >&2; exit 1; }
warn()  { echo -e "${YELLOW}WARNING: $1${NC}" >&2; }
info()  { echo -e "${GREEN}INFO: $1${NC}" >&2; }

check_pcluster() {
    command -v pcluster &>/dev/null || error "pcluster CLI not found. Install AWS ParallelCluster CLI first."
    info "pcluster CLI found: $(command -v pcluster)"
}

validate_ami_id() {
    [[ "$1" =~ ^ami-[a-f0-9]{8,17}$ ]] || error "Invalid AMI ID format: $1"
}

generate_config() {
    CONFIG_FILE=$(mktemp /tmp/pcluster-build-config-XXXXXX.yaml)
    cat > "${CONFIG_FILE}" << EOF
Build:
  InstanceType: ${INSTANCE_TYPE:-g5.xlarge}
  ParentImage: ${AMI_ID}
  UpdateOsPackages:
    Enabled: false
EOF
    info "Build configuration:"
    cat "${CONFIG_FILE}" >&2
}

cleanup() {
    [[ -n "${CONFIG_FILE}" && -f "${CONFIG_FILE}" ]] && rm -f "${CONFIG_FILE}"
}

start_build() {
    local build_name
    build_name="${IMAGE_NAME:-pcluster-centos9-efa-source-gpu-$(date +%Y%m%d-%H%M%S)}"

    info "Starting build: ${build_name}"
    info "Parent AMI: ${AMI_ID}"
    info "Region: ${REGION}"

    pcluster build-image \
        --image-id "${build_name}" \
        --image-configuration "${CONFIG_FILE}" \
        --region "${REGION}" >&2 || error "Failed to start pcluster build-image"

    echo "${build_name}"
}

poll_build_status() {
    local build_name="$1"
    info "Monitoring build (polling every ${POLL_INTERVAL}s)..."

    while true; do
        local output status ami_id
        output=$(pcluster describe-image --image-id "${build_name}" --region "${REGION}" 2>&1) \
            || error "Failed to get build status for ${build_name}"

        status=$(echo "${output}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('imageBuildStatus','UNKNOWN'))" 2>/dev/null)

        echo "STATUS: ${status}" >&2

        case "${status}" in
            BUILD_IN_PROGRESS) sleep "${POLL_INTERVAL}" ;;
            BUILD_COMPLETE)
                ami_id=$(echo "${output}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('ec2AmiInfo',{}).get('amiId',''))" 2>/dev/null)
                info "Build successful!"
                echo ""
                echo "ParallelCluster AMI: ${ami_id}"
                echo ""
                echo "Use in cluster config:"
                echo "  Image:"
                echo "    CustomAmi: ${ami_id}"
                return 0
                ;;
            BUILD_FAILED)
                error "Build failed! Check logs: pcluster get-image-log-events --image-id ${build_name} --region ${REGION} --log-stream-name 3.13.2/1"
                ;;
            *) warn "Status: ${status}. Continuing..."; sleep "${POLL_INTERVAL}" ;;
        esac
    done
}

parse_args() {
    [[ $# -eq 0 ]] && { usage; exit 1; }
    case "$1" in
        -h|--help) usage; exit 0 ;;
        -*) error "First argument must be an AMI ID." ;;
        *) AMI_ID="$1"; shift ;;
    esac
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --region)        REGION="${2:?--region requires a value}"; shift 2 ;;
            --image-name)    IMAGE_NAME="${2:?--image-name requires a value}"; shift 2 ;;
            --instance-type) INSTANCE_TYPE="${2:?--instance-type requires a value}"; shift 2 ;;
            -h|--help)       usage; exit 0 ;;
            *)               error "Unknown option: $1" ;;
        esac
    done
}

main() {
    parse_args "$@"
    validate_ami_id "${AMI_ID}"
    check_pcluster
    trap cleanup EXIT
    generate_config
    local build_name
    build_name=$(start_build)
    poll_build_status "${build_name}"
}

main "$@"
