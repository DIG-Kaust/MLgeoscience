import torch.nn as nn


def init_weights(model):
    """Initialize model weights

    Parameters
    ----------
    model : :obj:`torch.nn.Module`
        Model

    """
    if type(model) == nn.Linear:
        nn.init.xavier_uniform_(model.weight)
        if model.bias is not None:
            model.bias.data.fill_(0.01)