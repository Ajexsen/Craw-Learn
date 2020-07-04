#!/bin/sh
#source ~/anaconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate Fedder_RL
cd src
python3 main-optuna.py ${1}
