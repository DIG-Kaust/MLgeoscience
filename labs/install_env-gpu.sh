#!/bin/bash
# 
# Installer for ML course with GPU
# 
# Run: ./install_env.sh
# 
# M. Ravasi, 20/04/2021

echo 'Creating ML Course environment with Pytorch (GPU)'

# create conda env
conda env create -f environment-gpu.yml
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlcourse_gpu
conda env list
echo 'Created and activated environment:' $(which python)

# check torch works as expected
echo 'Checking torch version and running a command...'
python -c 'import torch; print(torch.__version__);  print(torch.cuda.get_device_name(torch.cuda.current_device())); print(torch.ones(10).to("cuda:0"))'

echo 'Done!'

