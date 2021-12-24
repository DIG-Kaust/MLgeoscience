import random
import numpy as np
import torch
import torch.nn as nn
import wandb
from torch.utils.data import TensorDataset, DataLoader


def create_dataloader(X, Z, xs, zs, v, tana, tana_dx, tana_dz,
                      batch_size=None, perc=0.25, shuffle=True, device='cpu'):
    XZ = torch.from_numpy(np.vstack((X, Z)).T).float().to(device)
    v = torch.from_numpy(v).float().to(device)
    tana = torch.from_numpy(tana).float().to(device)
    tana_dx = torch.from_numpy(tana_dx).float().to(device)
    tana_dz = torch.from_numpy(tana_dz).float().to(device)

    # select small number of random samples to be used as training data
    nxz = len(XZ)
    npoints = int(nxz * perc)
    if batch_size is None:
        batch_size = npoints
    if perc == 1.:
        ipermute = np.arange(nxz)
    else:
        ipermute = np.random.permutation(np.arange(nxz))[:npoints]
    dataset = TensorDataset(XZ[ipermute], v[ipermute],
                            tana[ipermute], tana_dx[ipermute], tana_dz[ipermute])
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # initial condition
    ic = torch.tensor([xs, zs], dtype=torch.float, requires_grad=True)

    return data_loader, ic


def create_gridloader(X, Z, device='cpu'):
    XZ = torch.from_numpy(np.vstack((X, Z)).T).float().to(device)
    grid = TensorDataset(XZ)
    grid_loader = DataLoader(grid, batch_size=XZ.size(0), shuffle=False)
    return grid_loader


def train(model, optimizer, data_loader, ic, lossweights=(1, 1), vscaler=1.):
    model.train()
    loss_pde = []
    loss_ic = []
    loss = []
    for xz, v, t0, t0_dx, t0_dz in data_loader:
        def closure():
            optimizer.zero_grad()

            # concatenate initial condition
            xz.requires_grad = True
            xzic = torch.cat([xz, ic.view(1, -1)])
            # compute tau
            tau = model(xzic).view(-1)

            # gradients
            gradient = torch.autograd.grad(tau, xzic, torch.ones_like(tau),
                                           create_graph=True)[0]
            tau_dx = gradient[:-1, 0]
            tau_dz = gradient[:-1, 1]

            # pde loss
            pde1 = (t0 ** 2) * (tau_dx ** 2 + tau_dz ** 2)
            pde2 = (tau[:-1] ** 2) * (t0_dx ** 2 + t0_dz ** 2)
            pde3 = 2 * t0 * tau[:-1] * (tau_dx * t0_dx + tau_dz * t0_dz)
            pde = (pde1 + pde2 + pde3) * vscaler - vscaler / (v ** 2)
            ls_pde = torch.mean(pde ** 2)

            # initial condition loss
            ls_ic = torch.mean((tau[-1] - 1) ** 2)

            # total Loss function:
            if model.lay == 'adaptive':
                local_recovery_terms = torch.tensor(
                    [torch.mean(model.model[layer][0].A.data) for layer in
                     range(len(model.model) - 1)])
                slope_recovery_term = 1. / torch.mean(torch.exp(local_recovery_terms))
                ls = lossweights[0] * ls_pde + lossweights[1] * ls_ic + \
                     slope_recovery_term
            else:
                ls = lossweights[0] * ls_pde + lossweights[1] * ls_ic
            loss_pde.append(ls_pde.item())
            loss_ic.append(ls_ic.item())
            loss.append(ls.item())
            # ls.backward(retain_graph = True)
            ls.backward()
            return ls
        optimizer.step(closure)

    loss_pde = np.sum(loss_pde) / len(data_loader)
    loss_ic = np.sum(loss_ic) / len(data_loader)
    loss = np.sum(loss) / len(data_loader)
    return loss, loss_pde, loss_ic


def evaluate(model, grid_loader):
    model.eval()
    with torch.no_grad():
        xz = iter(grid_loader).next()[0]
        # compute tau
        tau = model(xz)
    return tau


def evaluate_pde(model, grid_loader, vscaler=1.):
    model.train()
    xz, v, t0, t0_dx, t0_dz = iter(grid_loader).next()
    xz.requires_grad = True
    # compute tau
    tau = model(xz).view(-1)

    # gradients
    gradient = \
    torch.autograd.grad(tau, xz, torch.ones_like(tau), create_graph=True)[0]
    tau_dx = gradient[:, 0]
    tau_dz = gradient[:, 1]

    # pde
    pde1 = (t0 ** 2) * (tau_dx ** 2 + tau_dz ** 2)
    pde2 = (tau ** 2) * (t0_dx ** 2 + t0_dz ** 2)
    pde3 = 2 * t0 * tau * (tau_dx * t0_dx + tau_dz * t0_dz)
    pde_lhs = (pde1 + pde2 + pde3) * vscaler
    pde = pde_lhs - vscaler / (v ** 2)
    vpred = torch.sqrt(vscaler / pde_lhs)
    return pde, vpred


def training_loop(X, Z, xs, zs, v, tana, tana_dx, tana_dz,
                  model, optimizer, epochs,
                  randompoints=False, Xgrid=None, Zgrid=None,
                  batch_size=None, perc=0.25, lossweights=(1., 1.),
                  vscaler= 1., device='cpu'):
    if Xgrid is not None and Zgrid is not None:
        # Create gridloader
        grid_loader = create_gridloader(Xgrid, Zgrid, device)

    # Create dataloader
    data_loader, ic = create_dataloader(X, Z, xs, zs, v, tana, tana_dx, tana_dz,
                                        batch_size=batch_size, perc=perc, device=device)
    print('Number of points used per epoch:%d' % int(perc * len(X)))

    tau_history = []
    loss_pde_history = []
    loss_ic_history = []
    loss_history = []

    # Evaluate grid with randomly initialized network
    if Xgrid is not None and Zgrid is not None:
        tau_history.append(evaluate(model, grid_loader))

    for i in range(epochs):
        if randompoints and i > 0:
            # Recreate a new dataloader

            data_loader, ic = create_dataloader(X, Z, xs, zs, v, tana, tana_dx,
                                                tana_dz, batch_size=batch_size,
                                                perc=perc, device=device)
            if i == 1:
                print('Recreate dataloader...')
                print('Number of points used per epoch:%d' % int(perc * len(X)))

        # Train step
        loss, loss_pde, loss_ic = train(model, optimizer, data_loader, ic,
                                        lossweights=lossweights, vscaler=vscaler)
        loss_pde_history.append(loss_pde)
        loss_ic_history.append(loss_ic)
        loss_history.append(loss)

        # Store train loss in wandb logger
        try:
            wandb.log({"train_loss": loss})
        except:
            pass

        if i % 100 == 0:
            # Evaluate grid
            if Xgrid is not None and Zgrid is not None:
                tau_history.append(evaluate(model, grid_loader))
            print(f'Epoch {i}, Loss {loss:.7f}')

    return loss_history, loss_pde_history, loss_ic_history, tau_history