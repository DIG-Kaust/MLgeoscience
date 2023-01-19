The repository is organized as follows:

- **Data**: input data used in the practical sessions:

- **Notebooks**: Python codes and Jupyter Notebooks used in the practical sessions. More specifically,

   - **VisualOptimization**: Gradient-based optimization of a 2D cost function with visualization of model trajectories.
   - **BasicPytorch**: Introduction to PyTorch, Linear and Logistic Regression.
   - **LearningFunction**: A practical look at the Universal Approximation Theorem and its implications for learning functions with NNs.
   - **MixtureDensityNetwork**: Mixture Density Networks for multi-modal probabilistic predictions
   - **WellLogFaciesPrediction**: Well Log facies prediction based on Force2020 Challange
   - **EventDetection**: Seismic event detection with Recurrent NNs
   - **SaltNet**: U-Net Salt Segmentation based on TGS Kaggle Challange
   - **EikonalPINN**: Physics-informed NN solution of Eikonal equation

- **Wandb**: Example usage of [Wandb](https://wandb.ai/site) with Logistic Regression for efficient ML experimentation

## Environment (for All)

To ensure reproducibility of the results, we have provided a `environment.yml` file. Ensure to have installed Anaconda
or Miniconda on your computer. If you are not familiar with it, we suggesting using the 
[KAUST Miniconda Install recipe](https://github.com/kaust-rccl/ibex-miniconda-install). This has been tested 
both on macOS and Unix operative systems.

After that simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go!


## Environment (for KAUST Students)

For KAUST Students officially attending the course, from this year we will be providing you directly with the computational resources
needed for homeworks and projects. You do not need to worry about setting up an environment locally. 

Simply head over the the [ClassHub](https://classhub.kaust.edu.sa) platform from any browser and select our course. A remote JupyterHub
desktop will become available to you in seconds and you can work as if you were on your local laptop or workstation for the entire 
duration of the course.

## Environment (for KAUST Students - IBEX, Deprecated)

Later in the course, it may be useful to have access to a workstation with GPU capabilities (it will speed up your training time).
A modified version of the environment and installation files for GPU-powered environment are also provided here.

Various options exist to access a GPU. If you have a personal machine with a GPU you are lucky, 
take advantage of it. Alternatively, the [KAUST Ibex](https://www.hpc.kaust.edu.sa/ibex) cluster provides a large pool of nodes with different 
families of GPUs (RTX 2080, P6000, P100, V100). To install the GPU environment follow the following steps:
```
ssh ${USER}@glogin.ibex.kaust.edu.sa
salloc --time=01:00:00 --gres=gpu:v100:1 
srun --pty bash
./install_env-gpu.sh
```

A sample [SLURM file](https://github.com/DIG-Kaust/MLgeoscience/blob/main/labs/jupyter_notebook_ibex.slurm) is provided 
that allows setting up a Jupyter notebook with GPU capabilities. More details can be found [here](https://kaust-supercomputing-lab.atlassian.net/wiki/spaces/Doc/pages/88080449/Interactive+computing+using+Jupyter+Notebooks+on+KSL+platforms). 


