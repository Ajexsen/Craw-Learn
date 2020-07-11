#!/bin/sh
for i in `seq 1 3`;
do
    sbatch --partition=Antarktis,Luna,Sibirien --job-name=ppo_${i} --cpus-per-task=4 --output=slurm/slurm${i}.out ppo_maike.sh ${i}
done
squeue -u friedrichma
squeue -u friedrichma -h -t pending,running -r | wc -l
