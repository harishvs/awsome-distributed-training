docker build  -t fsdp:pytorch2.7.1 .
enroot import -o pytorch-fsdp.sqsh  dockerd://fsdp:pytorch2.7.1
mv pytorch-fsdp.sqsh slurm