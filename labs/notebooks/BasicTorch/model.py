import torch.nn as nn


class SingleHiddenLayerNetwork(nn.Module):
    """Single Hidden layer neural network

    Parameters
    ----------
    I : :obj:`int`
        Size of input layer
    H : :obj:`int`
        Size of hidden layer
    O : :obj:`int`
        Size of output layer

    """
    def __init__(self, I, H, O):
        super(SingleHiddenLayerNetwork, self).__init__()
        self.hidden_1 = nn.Linear(I, H, bias=True)
        self.output = nn.Linear(H, O, bias=True)
        self.sigmoid = nn.Sigmoid()
        # Add relu
        self.relu = nn.ReLU()

    def forward(self, x):
        z1 = self.hidden_1(x)
        a1 = self.relu(z1)  # use relu
        z2 = self.output(a1)
        a2 = self.sigmoid(z2)
        return a2