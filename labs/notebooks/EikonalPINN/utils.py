import random
import numpy as np
import torch
import skfmm


def set_seed(seed):
    """Set all random seeds to a fixed value and take out any randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled   = False


def set_device():
    device = 'cpu'
    if torch.cuda.device_count() > 0 and torch.cuda.is_available():
        print("Cuda installed! Running on GPU!")
        device = torch.device(torch.cuda.current_device())
        print(f'Device: {device} {torch.cuda.get_device_name(device)}')
    else:
        print("No GPU available!")
    return device


def eikonal_grid(ox, dx, nx, oz, dz, nz):
    x = np.arange(nx) * dx + ox
    z = np.arange(nz) * dz + oz

    X, Z = np.meshgrid(x, z, indexing='ij')
    X, Z = X.ravel(), Z.ravel()

    return x, z, X, Z


def eikonal_constant(ox, dx, nx, oz, dz, nz, xs, zs, v):
    """Eikonal solution in constant velocity

    Compute analytical eikonal solution and its spatial derivatives
    for constant velocity model

    """
    x, z, X, Z = eikonal_grid(ox, dx, nx, oz, dz, nz)
    nx, nz = len(x), len(z)

    # Analytical solution
    dana = np.sqrt((X - xs) ** 2 + (Z - zs) ** 2)
    tana = dana / v
    tana = tana.reshape(nx, nz)

    # Derivatives of analytical solution
    tana_dx = (X - xs) / (dana.ravel() * v)
    tana_dz = (Z - zs) / (dana.ravel() * v)
    tana_dx = tana_dx.reshape(nx, nz)
    tana_dz = tana_dz.reshape(nx, nz)

    return tana, tana_dx, tana_dz


def eikonal_gradient(ox, dx, nx, oz, dz, nz, xs, zs, v0, k):
    """Eikonal solution in gradient velocity

    Compute analytical eikonal solution for gradient velocity model

    """
    x, z, X, Z = eikonal_grid(ox, dx, nx, oz, dz, nz)
    nx, nz = len(x), len(z)

    # Velocity
    v = v0 + k * z
    vs = v[z == zs]
    v = np.outer(v, np.ones(nx)).T
    # Analytical solution
    dist2 = (X - xs) ** 2 + (Z - zs) ** 2
    tana = (1. / k) * np.arccosh(1 + (k**2 * dist2) / (2 * v.ravel() * vs))
    tana = tana.reshape(nx, nz)

    return tana


def eikonal_fmm(ox, dx, nx, oz, dz, nz, xs, zs, v):
    """Fast-marching method eikonal solution

    Compute eikonal solution using the fast-marching method for benchmark

    """
    x, z, X, Z = eikonal_grid(ox, dx, nx, oz, dz, nz)
    nx, nz = len(x), len(z)

    phi = np.ones((nx, nz))
    phi[int(xs // dx), int(zs // dz)] = -1.
    teik = skfmm.travel_time(phi, v, dx=(dx, dz))

    return teik


def remove_source(X, Z, xs, zs, v, tana, tana_dx, tana_dz):
    """Remove source from grids

    Remove element corresponding to the source index due to the fact that
    the analytical derivatives for the traveltime are undefined

    """
    # Find source index
    isource = (X.ravel() == xs) & (Z.ravel() == zs)

    X_nosrc, Z_nosrc = X.ravel()[~isource], Z.ravel()[~isource]
    v_nosrc, tana_nosrc = v.ravel()[~isource], tana.ravel()[~isource]
    tana_dx_nosrc, tana_dz_nosrc = tana_dx.ravel()[~isource], tana_dz.ravel()[~isource]

    return X_nosrc, Z_nosrc, v_nosrc, tana_nosrc, tana_dx_nosrc, tana_dz_nosrc