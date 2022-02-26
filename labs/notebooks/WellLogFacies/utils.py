import random
import numpy as np
import torch


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