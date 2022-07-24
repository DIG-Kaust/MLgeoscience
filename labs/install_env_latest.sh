#!/bin/bash
# 
# Installer for ML course
# 
# Run: ./install_env_latest.sh
# 
# M. Ravasi, 20/04/2021

echo 'Creating ML Course environment with latest Pytorch (CPU)'

# create conda env
conda env create -f environment_latest.yml
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlcourse_latest
conda env list
echo 'Created and activated environment:' $(which python)

# check torch works as expected
echo 'Checking torch version and running a command...'
python -c 'import torch; print(torch.__version__); print(torch.ones(10))'

echo 'Done!'

