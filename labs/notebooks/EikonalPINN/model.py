import random
import numpy as np
import torch
import torch.nn as nn


class Swish(nn.Module):
    r"""Applies the element-wise function:
    .. math::
        \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """

    def forward(self, input):
        return torch.sigmoid(input) * input


def activation(act_fun='LeakyReLU'):
    """Easy selection of activation function by passing string or
    module (e.g. nn.ReLU)
    """
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'ReLU':
            return nn.ReLU()
        elif act_fun == 'Tanh':
            return nn.Tanh()
        elif act_fun == 'Swish':
            return Swish()
        else:
            raise ValueError(f'{act_fun} is not an activation function...')
    else:
        return act_fun()
    
    
class AdaptiveLinear(nn.Linear):
    r"""Applies a linear transformation to the input data as follows
    :math:`y = naxA^T + b`.

    From https://github.com/antelk/locally-adaptive-activation-functions.
    More details available in Jagtap, A. D. et al. Locally adaptive
    activation functions with slope recovery for deep and
    physics-informed neural networks, Proc. R. Soc. 2020.

    Parameters
    ----------
    in_features : int
        The size of each input sample
    out_features : int
        The size of each output sample
    bias : bool, optional
        If set to ``False``, the layer will not learn an additive bias
    adaptive_rate : float, optional
        Scalable adaptive rate parameter for activation function that
        is added layer-wise for each neuron separately. It is treated
        as learnable parameter and will be optimized using a optimizer
        of choice
    adaptive_rate_scaler : float, optional
        Fixed, pre-defined, scaling factor for adaptive activation
        functions

    """
    def __init__(self, in_features, out_features, bias=True,
                 adaptive_rate=None, adaptive_rate_scaler=None):
        super(AdaptiveLinear, self).__init__(in_features, out_features, bias)
        self.adaptive_rate = adaptive_rate
        self.adaptive_rate_scaler = adaptive_rate_scaler
        if self.adaptive_rate:
            self.A = nn.Parameter(self.adaptive_rate * torch.ones(self.in_features))
            if not self.adaptive_rate_scaler:
                self.adaptive_rate_scaler = 10.0

    def forward(self, input):
        if self.adaptive_rate:
            return nn.functional.linear(
                self.adaptive_rate_scaler * self.A * input, self.weight,
                self.bias)
        return nn.functional.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return (
            f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, '
            f'adaptive_rate={self.adaptive_rate is not None}, adaptive_rate_scaler={self.adaptive_rate_scaler is not None}'
        )


def layer(lay='linear'):
    """Easy selection of layer
    """
    if isinstance(lay, str):
        if lay == 'linear':
            return lambda x,y: nn.Linear(x, y)
        elif lay == 'adaptive':
            return lambda x,y: AdaptiveLinear(x,y,
                                              adaptive_rate=0.1,
                                              adaptive_rate_scaler=10.)
        else:
            raise ValueError(f'{lay} is not a layer type...')
    else:
        return lay


class Network(nn.Module):
    def __init__(self, n_input, n_output, n_hidden=[16, 32],
                 lay='linear', act='Tanh'):
        super(Network, self).__init__()
        self.lay = lay
        self.act = act
        act = activation(act)
        lay = layer(lay)
        self.model = nn.Sequential(nn.Sequential(lay(n_input, n_hidden[0]), act),
                                   *[nn.Sequential(lay(n_hidden[i], n_hidden[i + 1]),
                                     act) for i in range(len(n_hidden) - 1)],
                                   lay(n_hidden[-1], n_output))

    def forward(self, x):
        x = self.model(x)
        return x

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)