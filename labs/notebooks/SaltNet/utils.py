import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def set_seed(seed):
    """Set all random seeds to a fixed value and take out any
    randomness from cuda kernels

    Parameters
    ----------
    seed : :obj:`int`
        Seed number

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    return True


def show_tensor_images(image_tensor, ax, num_images=15, vmin=-1, vmax=1, cbar=False):
    """Visualize images

    Given a tensor of images, display a portion of them aligned in a single figure

    Parameters
    ----------
    image_tensor : :obj:`torch.Tensor`
        Torch tensor of size nbatch x nch x nh x nw
    ax : :obj:`matplotlib.axes`
        Figure axis where to display a figure
    num_images : :obj:`int`, optional
        Number of images to display
    vmin : :obj:`float`, optional
        Minimum value in colorscale
    vmax : :obj:`float`, optional
        Maximum value in colorscale
    cbar : :obj:`bool`, optional
        Display colorbar

    """
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5, normalize=False)
    ax.axis('off')
    im = ax.imshow(image_grid[0].squeeze(), cmap='gray', vmin=vmin, vmax=vmax)
    if cbar:
        plt.colorbar(im, ax=ax)