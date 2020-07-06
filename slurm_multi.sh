#!/bin/sh
eval "$(conda shell.bash hook)"
conda activate Fedder_RL
cd src
python3 -c "import optuna;optuna.study.delete_study(study_name='crawler-JR', storage='sqlite:///example.db')"
python3 -c "import optuna;optuna.study.create_study(study_name='crawler-JR', storage='sqlite:///example.db')"
cd ..
#optuna delete-study 'crawler-JR'
#optuna dashboard --study-name "crawler-JR" --storage "sqlite:///example.db"
for i in `seq 1 24`;
do
    sbatch --partition=Luna,Sibirien,Antarktis --job-name=ppo_${i} --cpus-per-task=4 --output=slurm/slurm${i}.out ppo_fed.sh ${i}
done
squeue -u jensenj
squeue -u jensenj -h -t pending,running -r | wc -l
