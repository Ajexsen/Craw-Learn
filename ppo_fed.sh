#!/bin/sh
#git clone https://github.com/dbarnett/python-helloworld.git
#cd python-helloworld
#source ~/anaconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate Fedder_RL
python3 src/main.py
