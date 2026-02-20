#!/bin/bash
#
# create-cluster.sh
#
# Create a ParallelCluster using a custom CentOS 9 EFA AMI.
# Since the AMI already has EFA/NCCL/libfabric built from source,
# the post-install NCCL scripts are skipped.
#
# Usage:
#   ./create-cluster.sh <ami-id> [options]
#

set -euo pipefail

# Defaults
REGION="${AWS_REGION:-us-east-1}"
CLUSTER_NAME=""
INSTANCE_TYPE="${INSTANCE:-g5.8xlarge}"
NUM_INSTANCES="${NUM_INSTANCES:-2}"
KEY_PAIR="${KEY_PAIR_NAME:-harish_kp}"
STACK_ID="${STACK_ID_VPC:-parallelcluster-prerequisites-centos9-cluster}"
CONFIG_FILE=""

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

usage() {
    cat << EOF
Usage: $(basename "$0") <ami-id> [options]

Create a ParallelCluster with a custom CentOS 9 EFA AMI.

Arguments:
  ami-id                      Required. The ParallelCluster-ready AMI ID.

Options:
  --cluster-name <name>       Cluster name (default: centos9-ml-TIMESTAMP)
  --region <region>           AWS region (default: ${REGION})
  --instance-type <type>      Compute instance type (default: ${INSTANCE_TYPE})
  --num-instances <n>         Number of compute nodes (default: ${NUM_INSTANCES})
  --key-pair <name>           EC2 key pair name for SSH
  --stack-name <name>         VPC CloudFormation stack name (default: ${STACK_ID})
  -h, --help                  Show this help message

Environment variables (override defaults):
  AWS_REGION, INSTANCE, NUM_INSTANCES, KEY_PAIR_NAME, STACK_ID_VPC

Prerequisites:
  Deploy the VPC stack from 1.architectures/1.vpc_network/ first.
EOF
}

error() { echo -e "${RED}ERROR: $1${NC}" >&2; exit 1; }
info()  { echo -e "${GREEN}INFO: $1${NC}" >&2; }

get_stack_output() {
    local key="$1"
    aws cloudformation describe-stacks \
        --stack-name "${STACK_ID}" \
        --query "Stacks[0].Outputs[?OutputKey==\`${key}\`].OutputValue" \
        --region "${REGION}" \
        --output text 2>/dev/null
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
            --cluster-name)  CLUSTER_NAME="${2:?requires value}"; shift 2 ;;
            --region)        REGION="${2:?requires value}"; shift 2 ;;
            --instance-type) INSTANCE_TYPE="${2:?requires value}"; shift 2 ;;
            --num-instances) NUM_INSTANCES="${2:?requires value}"; shift 2 ;;
            --key-pair)      KEY_PAIR="${2:?requires value}"; shift 2 ;;
            --stack-name)    STACK_ID="${2:?requires value}"; shift 2 ;;
            -h|--help)       usage; exit 0 ;;
            *)               error "Unknown option: $1" ;;
        esac
    done
    CLUSTER_NAME="${CLUSTER_NAME:-centos9-ml-$(date +%Y%m%d-%H%M%S)}"
}

lookup_infra() {
    info "Looking up infrastructure from stack: ${STACK_ID}"

    PUBLIC_SUBNET=$(get_stack_output PublicSubnet)
    [[ -z "${PUBLIC_SUBNET}" ]] && error "Could not find PublicSubnet in stack ${STACK_ID}"

    PRIVATE_SUBNET=$(get_stack_output PrimaryPrivateSubnet)
    [[ -z "${PRIVATE_SUBNET}" ]] && PRIVATE_SUBNET=$(get_stack_output PrivateSubnet)
    [[ -z "${PRIVATE_SUBNET}" ]] && error "Could not find private subnet in stack ${STACK_ID}"

    SECURITY_GROUP=$(get_stack_output SecurityGroup)
    [[ -z "${SECURITY_GROUP}" ]] && error "Could not find SecurityGroup in stack ${STACK_ID}"

    info "Public subnet:  ${PUBLIC_SUBNET}"
    info "Private subnet: ${PRIVATE_SUBNET}"
    info "Security group: ${SECURITY_GROUP}"
}

generate_config() {
    CONFIG_FILE=$(mktemp /tmp/pcluster-config-XXXXXXXX)

    local ssh_block=""
    if [[ -n "${KEY_PAIR}" ]]; then
        ssh_block="  Ssh:
    KeyName: ${KEY_PAIR}"
    fi

    cat > "${CONFIG_FILE}" << EOF
Region: ${REGION}
Imds:
  ImdsSupport: v2.0
Image:
  Os: rhel9
  CustomAmi: ${AMI_ID}
HeadNode:
  InstanceType: m5.8xlarge
  Networking:
    SubnetId: ${PUBLIC_SUBNET}
    AdditionalSecurityGroups:
      - ${SECURITY_GROUP}
${ssh_block}
  LocalStorage:
    RootVolume:
      Size: 500
      DeleteOnTermination: true
  Iam:
    AdditionalIamPolicies:
      - Policy: arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
      - Policy: arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
  Imds:
    Secured: false
Scheduling:
  Scheduler: slurm
  SlurmSettings:
    ScaledownIdletime: 60
    QueueUpdateStrategy: DRAIN
  SlurmQueues:
    - Name: compute-gpu
      CapacityType: ONDEMAND
      Networking:
        SubnetIds:
          - ${PRIVATE_SUBNET}
        PlacementGroup:
          Enabled: true
        AdditionalSecurityGroups:
          - ${SECURITY_GROUP}
      ComputeSettings:
        LocalStorage:
          EphemeralVolume:
            MountDir: /scratch
          RootVolume:
            Size: 512
      ComputeResources:
        - Name: distributed-ml
          InstanceType: ${INSTANCE_TYPE}
          MinCount: 0
          MaxCount: ${NUM_INSTANCES}
          Efa:
            Enabled: true
Monitoring:
  DetailedMonitoring: true
  Logs:
    CloudWatch:
      Enabled: true
  Dashboards:
    CloudWatch:
      Enabled: true
EOF

    info "Cluster config written to: ${CONFIG_FILE}"
}

main() {
    parse_args "$@"
    command -v pcluster &>/dev/null || error "pcluster CLI not found."

    lookup_infra
    generate_config

    info "Creating cluster: ${CLUSTER_NAME}"
    info "AMI: ${AMI_ID}"
    info "Instance type: ${INSTANCE_TYPE}"
    info "Max nodes: ${NUM_INSTANCES}"

    pcluster create-cluster \
        --cluster-name "${CLUSTER_NAME}" \
        --cluster-configuration "${CONFIG_FILE}" \
        --region "${REGION}"

    echo ""
    info "Cluster creation started. Monitor with:"
    echo "  pcluster describe-cluster --cluster-name ${CLUSTER_NAME} --region ${REGION}"
    echo ""
    echo "SSH into head node:"
    echo "  pcluster ssh --cluster-name ${CLUSTER_NAME} --region ${REGION}"
}

main "$@"
