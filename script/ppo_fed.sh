#!/bin/sh
#source ~/anaconda3/etc/profile.d/conda.sh
eval "$(conda shell.bash hook)"
conda activate Federer
chmod -R 755 crawler_single/linux/static_server/crawler_static.x86_64
cd src
python3 main2.py ${1}
