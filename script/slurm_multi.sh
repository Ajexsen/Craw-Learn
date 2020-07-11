#!/bin/sh
eval "$(conda shell.bash hook)"
conda activate Federer
cd src
#python3 -c "import optuna;optuna.study.delete_study(study_name='crawler-JR-ppo1', storage='sqlite:///example.db')"
#python3 -c "import optuna;optuna.study.create_study(study_name='crawler-JR-ppo1', storage='sqlite:///example.db')"
cd ..
#optuna delete-study 'crawler-JR'
#optuna dashboard --study-name "crawler-JR" --storage "sqlite:///example.db"
for i in `seq 160 175`;
do
#    sbatch --partition=Luna,Sibirien,Gobi --job-name=ppo1_${i} --cpus-per-task=2 --output=slurm/slurm${i}.out ppo_fed.sh ${i}
#    sbatch --partition=Luna,Sibirien,Gobi --job-name=ppo_std_${i} --cpus-per-task=1 --output=slurm/slurm_std_${i}.out ppo_fed.sh ${i}
    sbatch --partition=All --job-name=ppo_std_${i} --cpus-per-task=4 --output=slurm/slurm_std_${i}.out ppo_fed.sh ${i}
done

squeue -u jensenj
squeue -u jensenj -h -t pending,running -r | wc -l
