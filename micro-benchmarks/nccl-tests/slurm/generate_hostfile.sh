#!/bin/bash

# Script get topologically sorted hostnames
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
hostname_file="hostnames.txt"

slurm_nodes=$(sinfo -N -h -o "%N" | sort -u | tee "$hostname_file")


