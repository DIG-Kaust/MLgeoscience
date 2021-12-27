# Machine Learning in Geoscience

Teaching material for ErSE 222 Machine Learning in Geoscience course to be held at KAUST during the Spring semester.

## Material

The repository is organized as follows:

- **Data**: input data used in the practical sessions:
- All of the other folders in this repository contains Python codes and Jupyter Notebooks used in the practical sessions:

- **Notebooks**: input data used in the practical sessions:

   - **VisualOptimization**: Gradient-based optimization of a 2D cost function with visualization of model trajectories.
   - **BasicPytorch**: Introduction to PyTorch, Linear and Logistic Regression.
   - **LearningFunction**: A practical look at the Universal Approximation Theorem and its implications for learning functions with NNs.
   - **MixtureDensityNetwork**: Mixture Density Networks for multi-modal probabilistic predictions
   - **WellLogFaciesPrediction**: Well Log facies prediction based on Force2020 Challange
   - **EventDetection**: Seismic event detection with Recurrent NNs
   - **SaltNet**: U-Net Salt Segmentation based on TGS Kaggle Challange
   - **EikonalPINN**: Physics-informed NN solution of Eikonal equation

- **Wandb**: example usage of [Wandb](https://wandb.ai/site) with Logistic Regression for efficient ML experimentation

## Environment

To ensure reproducibility of the results, we have provided a `environment.yml` file. Ensure to have installed Anaconda or Miniconda on your computer. If you are not familiar with it, we suggesting using the [KAUST Miniconda Install recipe](https://github.com/kaust-rccl/ibex-miniconda-install). This has been tested both on macOS and Unix operative systems.

After that simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the work `Done!` on your terminal you are ready to go!