#!/bin/bash
export OMPI_MCA_rmaps=seq
mpirun --host $SLURM_HOSTLIST "$@"
