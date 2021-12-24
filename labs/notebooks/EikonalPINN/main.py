#!/usr/bin/env python3
"""
Eikonal solver using PINNs with Wandb logging

"""
import argparse
import matplotlib.pyplot as plt
import torch
import wandb

from utils import *
from model import *
from train import *


def main(parser):

    ######### INPUT PARAMS #########
    parser.add_argument('-u', '--unit', type=int,  default=10, help='Size of hidden layer')
    parser.add_argument('-H', '--hidden', type=int,  default=10, help='Number of hidden layers')
    parser.add_argument('-a', '--act', type=str,  default='Tanh', help='Activation function')
    parser.add_argument('-l', '--learning_rate', type=float,  default=0.1, help='Learning rate')
    parser.add_argument('-e', '--nepochs', type=int,  default=100, help='Number of epochs')
    parser.add_argument('-p', '--lambda_pde', type=float, default=1., help='PDE weight')
    parser.add_argument('-i', '--lambda_ic', type=float, default=1., help='IC weight')
    parser.add_argument('-s', '--perc_samples', type=float, default=.25, help='Percentage of training samples '
                                                                              'over the entire grid')
    parser.add_argument('-w', '--wandb', default=False, action='store_true', help='Enable Wandb')

    args = parser.parse_args()

    if args.wandb:
        # Define wandb project to use
        wandb.init(project='eikonal-pinn')

        # Crete configure object to which we can attach the parameters to log
        config = wandb.config

        # Define experiment parameters to be logged
        config.hidden = args.hidden
        config.unit = args.unit
        config.act = args.act
        config.learning_rate = args.learning_rate
        config.nepochs = args.nepochs
        config.lambda_pde = args.lambda_pde
        config.lambda_ic = args.lambda_ic
        config.perc_samples = args.perc_samples

    # Setup environment
    set_seed(10)
    device = set_device()

    # Model grid (km)
    ox, dx, nx = 0., 10. / 1000., 101
    oz, dz, nz = 0., 10. / 1000., 201

    # Velocity model (km/s)
    v0 = 1000. / 1000.
    vel = v0 * np.ones((nx, nz))

    # Source (km)
    xs, zs = 500. / 1000., 500. / 1000.

    # Computational domain
    x, z, X, Z = eikonal_grid(ox, dx, nx, oz, dz, nz)

    # Analytical solution
    tana, tana_dx, tana_dz = eikonal_constant(ox, dx, nx, oz, dz, nz,
                                              xs, zs, v0)
    tana_dx_numerical, tana_dz_numerical = np.gradient(tana, dx, dz)

    # Eikonal solution
    teik = eikonal_fmm(ox, dx, nx, oz, dz, nz, xs, zs, vel)

    # Factorized eikonal solution: t= tau * t0
    tau = teik / tana

    # Remove source from grid of points to be used in training
    X_nosrc, Z_nosrc, v_nosrc, tana_nosrc, tana_dx_nosrc, tana_dz_nosrc = \
        remove_source(X, Z, xs, zs, vel, tana, tana_dx, tana_dz)

    # Create evaluation grid
    grid_loader = create_gridloader(X, Z, device)

    # Define and initialize network
    model = Network(2, 1, [args.unit] * args.hidden, act=args.act)
    model.to(device)
    # model.apply(model.init_weights)
    print(model)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 betas=(0.9, 0.999), eps=1e-5)

    # Training
    loss_history, loss_pde_history, loss_ic_history, tau_history = \
        training_loop(X_nosrc, Z_nosrc, xs, zs, v_nosrc, tana_nosrc,
                      tana_dx_nosrc, tana_dz_nosrc,
                      model, optimizer, args.nepochs, Xgrid=X, Zgrid=Z,
                      randompoints=False, batch_size=None, perc=args.perc_samples,
                      lossweights=(args.lambda_pde, args.lambda_ic),
                      device=device)

    plt.figure()
    plt.semilogy(loss_history, 'k')

    # Compute traveltime with trained network
    tau_est = evaluate(model, grid_loader)

    # Compute traveltime error over grid
    error = np.linalg.norm(tana.ravel() -
                           (tau_est.detach().cpu().numpy().reshape(nx, nz) * tana).ravel())
    if args.wandb:
        wandb.log({"grid_error": error})

    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    im = axs[0].imshow(tau_est.detach().cpu().numpy().reshape(nx, nz),
                       vmin=0.7, vmax=1.3,
                       extent=(x[0], x[-1], z[0], z[-1]), cmap='gray_r',
                       origin='lower')
    axs[0].scatter(xs, zs, s=200, marker='*', color='k')
    axs[0].set_title('Tau')
    axs[0].axis('tight')
    plt.colorbar(im, ax=axs[0])

    im = axs[1].contour(tana.T, extent=(x[0], x[-1], z[0], z[-1]), colors='k',
                        label='Analytical')
    axs[1].contour((tau_est.detach().cpu().numpy().reshape(nx, nz) * tana).T,
                   extent=(x[0], x[-1], z[0], z[-1]), colors='r',
                   label='Estimated')
    axs[1].scatter(xs, zs, s=200, marker='*', color='k')
    axs[1].set_title('Traveltimes')
    axs[1].axis('tight')
    plt.colorbar(im, ax=axs[1])
    im = axs[2].imshow(
        tana.T - (tau_est.detach().cpu().numpy().reshape(nx, nz) * tana).T,
        vmin=-0.001, vmax=0.001,
        extent=(x[0], x[-1], z[0], z[-1]), cmap='jet', origin='lower')
    axs[2].scatter(xs, zs, s=200, marker='*', color='k')
    axs[2].set_title('Error')
    axs[2].axis('tight')
    plt.colorbar(im, ax=axs[2])

    if args.wandb:
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":

    description = 'Eikonal PINN with wandb'
    main(argparse.ArgumentParser(description=description))

# python main.py -u 100 -H 3 -a Tanh -l 0.001 -e 1000 -p 1. -i 10. -s 0.25
# python main.py -u 100 -H 3 -a Tanh -l 0.001 -e 1000 -p 1. -i 10. -s 0.1