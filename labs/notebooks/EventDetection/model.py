import torch.nn as nn


class LSTMNetwork(nn.Module):
    """LSTM network

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
        super(LSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(I, H, 1, batch_first=True)
        self.dense = nn.Linear(H, O, bias=False)

    def forward(self, x):
        z, _ = self.lstm(x)
        out = self.dense(z)
        return out.squeeze()


class BiLSTMNetwork(nn.Module):
    """Bidirectional-LSTM network

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
        super(BiLSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(I, H, 1, batch_first=True, bidirectional=True)
        self.dense = nn.Linear(2 * H, O, bias=False)

    def forward(self, x):
        z, _ = self.lstm(x)
        out = self.dense(z)
        return out.squeeze()


class DoubleBiLSTMNetwork(nn.Module):
    """Deep Bidirectional-LSTM network

    Parameters
    ----------
    I : :obj:`int`
        Size of input layer
    He : :obj:`int`
        Size of first hidden layer
    Hd : :obj:`int`
        Size of second hidden layer
    O : :obj:`int`
        Size of output layer

    """
    def __init__(self, I, He, Hd, O):
        super(DoubleBiLSTMNetwork, self).__init__()
        self.encoder = nn.LSTM(I, He, 1, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(2 * He, Hd, batch_first=True)
        self.dense = nn.Linear(Hd, O, bias=False)

    def forward(self, x):
        # Encoder
        z, _ = self.encoder(x)
        # Decoder
        z2, _ = self.decoder(z)
        # Dense
        out = self.dense(z2)
        return out.squeeze()