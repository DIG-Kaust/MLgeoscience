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
    parser.add_argument('-S', '--seed', type=int, default=10, help='Seed')
    parser.add_argument('-u', '--unit', type=int,  default=10, help='Size of hidden layer')
    parser.add_argument('-H', '--hidden', type=int,  default=10, help='Number of hidden layers')
    parser.add_argument('-a', '--act', type=str,  default='Tanh', help='Activation function')
    parser.add_argument('-l', '--lay', type=str, default='linear', help='Layer type')
    parser.add_argument('-o', '--optimizer', type=str,  default='adam', help='Optimizer')
    parser.add_argument('-r', '--learning_rate', type=float,  default=0.1, help='Learning rate')
    parser.add_argument('-e', '--nepochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('-s', '--perc_samples', type=float, default=.25,
                        help='Percentage of training samples over the entire grid')
    parser.add_argument('-R', '--randompoints', default=False, action='store_true',
                        help='Randomize point at each epoch')
    parser.add_argument('-b', '--batchsize', type=int, default=256, help='Batch size')
    parser.add_argument('-p', '--lambda_pde', type=float, default=1., help='PDE weight')
    parser.add_argument('-i', '--lambda_ic', type=float, default=1., help='IC weight')
    parser.add_argument('-w', '--wandb', default=False, action='store_true', help='Enable Wandb')
    parser.add_argument('-D', '--debug', default=False, action='store_true', help='Debug')

    args = parser.parse_args()

    print('Eikonal PINN')
    print('----------------------------')
    print(f'Seed = {args.seed}')
    print(f'Number of units = {args.unit}')
    print(f'Hidden size = {args.hidden}')
    print(f'Activation type = {args.act}')
    print(f'Layer type = {args.lay}')
    print(f'Optimizer = {args.optimizer}')
    print(f'Learning Rate = {args.learning_rate}')
    print(f'Number of epochs = {args.nepochs}')
    print(f'Percentage of Samples = {args.perc_samples}')
    print(f'Randomized selection per epochs = {args.randompoints}')
    print(f'Batch Size = {args.batchsize}')
    print(f'PDE Loss weight = {args.lambda_pde}')
    print(f'IC Loss weight = {args.lambda_ic}')
    print(f'Wandb = {args.wandb}')
    print(f'Debug = {args.debug}')
    print('----------------------------\n\n\n')

    args.batchsize = None if args.batchsize == 0 else args.batchsize

    if args.wandb:
        # Define wandb project to use
        wandb.init(project='eikonal-pinn')

        # Crete configure object to which we can attach the parameters to log
        config = wandb.config

        # Define experiment parameters to be logged
        config.hidden = args.hidden
        config.unit = args.unit
        config.act = args.act
        config.lay = args.lay
        config.learning_rate = args.learning_rate
        config.nepochs = args.nepochs
        config.lambda_pde = args.lambda_pde
        config.lambda_ic = args.lambda_ic
        config.perc_samples = args.perc_samples

    # Setup environment
    set_seed(args.seed)
    device = set_device()

    # Model grid (km)
    ox, dx, nx = 0., 10. / 1000., 101
    oz, dz, nz = 0., 10. / 1000., 201

    # Velocity model (km/s)
    v0 = 2000. / 1000.
    k = 0.5
    z = np.arange(nz) * dz + oz
    vel = np.outer((v0 + k * z), np.ones(nx)).T

    # Source (km)
    xs, zs = 500. / 1000., 500. / 1000.

    # Computational domain
    x, z, X, Z = eikonal_grid(ox, dx, nx, oz, dz, nz)

    # Analytical solution
    isource = (X == xs) & (Z == zs)
    vsource = vel.ravel()[isource][0]
    t0, t0_dx, t0_dz = eikonal_constant(ox, dx, nx, oz, dz, nz, xs, zs, vsource)
    tana = eikonal_gradient(ox, dx, nx, oz, dz, nz, xs, zs, v0, k)

    # Factorized eikonal solution: t= tau * t0
    tauana = tana / t0
    tauana[np.isnan(tauana)] = 1.

    # Remove source from grid of points to be used in training
    X_nosrc, Z_nosrc, v_nosrc, t0_nosrc, t0_dx_nosrc, t0_dz_nosrc = \
        remove_source(X, Z, xs, zs, vel, t0, t0_dx, t0_dz)

    # Create evaluation grid
    grid_loader = create_gridloader(X, Z, device)

    # Define and initialize network
    model = Network(2, 1, [args.unit] * args.hidden, act=args.act, lay=args.lay)
    model.to(device)
    # model.apply(model.init_weights)
    if args.debug:
        print(model)

    # Optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                     betas=(0.9, 0.999), eps=1e-5)
    elif args.optimizer == 'lbfgs':
        optimizer = torch.optim.LBFGS(model.parameters(),
                                      line_search_fn="strong_wolfe")

    # Training
    loss_history, loss_pde_history, loss_ic_history, tau_history = \
        training_loop(X_nosrc, Z_nosrc, xs, zs, v_nosrc,
                      t0_nosrc, t0_dx_nosrc, t0_dz_nosrc,
                      model, optimizer, args.nepochs, Xgrid=X, Zgrid=Z,
                      randompoints=args.randompoints, batch_size=args.batchsize,
                      perc=args.perc_samples, lossweights=(args.lambda_pde, args.lambda_ic),
                      device=device)

    plt.figure()
    plt.semilogy(loss_history, 'k')

    # Compute traveltime with trained network
    tau_est = evaluate(model, grid_loader)

    # Compute traveltime error over grid
    error = np.linalg.norm(tana.ravel() - (tau_est.detach().cpu().numpy().reshape(nx, nz) * t0).ravel())
    if args.wandb:
        wandb.log({"grid_error": error})

    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    im = axs[0].imshow(tau_est.detach().cpu().numpy().reshape(nx, nz).T,
                       vmin=0.7, vmax=1.3,
                       extent=(x[0], x[-1], z[0], z[-1]), cmap='jet',
                       origin='lower')
    axs[0].scatter(xs, zs, s=200, marker='*', color='k')
    axs[0].set_title('Tau')
    axs[0].axis('tight')
    plt.colorbar(im, ax=axs[0])

    im = axs[1].contour(tana.T, extent=(x[0], x[-1], z[0], z[-1]), colors='k',
                        label='Analytical')
    axs[1].contour((tau_est.detach().cpu().numpy().reshape(nx, nz) * t0).T,
                   extent=(x[0], x[-1], z[0], z[-1]), colors='r',
                   label='Estimated')
    axs[1].scatter(xs, zs, s=200, marker='*', color='k')
    axs[1].set_title('Traveltimes')
    axs[1].axis('tight')
    plt.colorbar(im, ax=axs[1])
    im = axs[2].imshow(tana.T - (tau_est.detach().cpu().numpy().reshape(nx, nz) * t0).T,
                       vmin=-0.001, vmax=0.001, extent=(x[0], x[-1], z[0], z[-1]),
                       cmap='jet', origin='lower')
    axs[2].scatter(xs, zs, s=200, marker='*', color='k')
    axs[2].set_title('Error')
    axs[2].axis('tight')
    plt.colorbar(im, ax=axs[2])

    if args.wandb or not args.debug:
        plt.close()
    else:
        plt.show()

    np.savez('Losses_grad_highvel/highgradient_unit%d_hidden_%d_lay%s_act%s_optimizer%s_epochs%d_lr%f_perc%.2f_lambda_init%d_batchsize%d_randompoints%d_seed%d'
        % (args.unit, args.hidden, args.lay, args.act, args.optimizer, args.nepochs,
           args.learning_rate, args.perc_samples, args.lambda_ic,
           0 if args.batchsize is None else args.batchsize, 1 if args.randompoints else 0, args.seed),
        loss=loss_history, error=error)

if __name__ == "__main__":

    description = 'Eikonal PINN with wandb'
    main(argparse.ArgumentParser(description=description))