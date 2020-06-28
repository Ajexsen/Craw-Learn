#!/bin/sh
eval "$(conda shell.bash hook)"
conda activate Fedder_RL
cd src
python3 -c "import optuna;optuna.study.delete_study(study_name='crawler-JR', storage='sqlite:///example.db')"
python3 -c "import optuna;optuna.study.create_study(study_name='crawler-JR', storage='sqlite:///example.db')"
#optuna delete-study 'crawler-JR'
cd ..
for i in `seq 1 25`;
do
    sbatch --partition=All --job-name=ppo_${i} --cpus-per-task=4 --output=slurm/slurm${i}.out ppo_fed.sh ${i}
done
squeue -u jensenj
squeue -u jensenj -h -t pending,running -r | wc -l
