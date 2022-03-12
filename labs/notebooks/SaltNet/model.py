import torch
import torch.nn as nn


def weights_init(m):
    """Initialize weights
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)


class ContractingBlock(nn.Module):
    """Contracting block

    Single block in contracting path composed of two convolutions followed by a max pool operation.
    We allow also to optionally include a batch normalization and dropout step.

    Parameters
    ----------
    input_channels : :obj:`int`
        Number of input channels
    use_dropout : :obj:`bool`, optional
        Add dropout
    use_bn : :obj:`bool`, optional
        Add batch normalization

    """
    def __init__(self, input_channels, use_dropout=False, use_bn=True):
        super(ContractingBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, input_channels * 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(input_channels * 2, input_channels * 2, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        if use_bn:
            self.batchnorm = nn.BatchNorm2d(input_channels * 2, momentum=0.8)
        self.use_bn = use_bn
        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.conv2(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.maxpool(x)
        return x


class ExpandingBlock(nn.Module):
    """Expanding block

    Single block in expanding path composed of an upsampling layer, a convolution, a concatenation of
    its output with the features at the same level in the contracting path, two additional convolutions.
    We allow also to optionally include a batch normalization and dropout step.

    Parameters
    ----------
    input_channels : :obj:`int`
        Number of input channels
    use_dropout : :obj:`bool`, optional
        Add dropout
    use_bn : :obj:`bool`, optional
        Add batch normalization

    """
    def __init__(self, input_channels, use_dropout=False, use_bn=True):
        super(ExpandingBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(input_channels, input_channels // 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(input_channels // 2, input_channels // 2, kernel_size=3, padding=1)
        if use_bn:
            self.batchnorm = nn.BatchNorm2d(input_channels // 2, momentum=0.8)
        self.use_bn = use_bn
        self.activation = nn.ReLU()
        if use_dropout:
            self.dropout = nn.Dropout()
        self.use_dropout = use_dropout

    def forward(self, x, skip_con_x):
        x = self.upsample(x)
        x = self.conv1(x)
        x = torch.cat([x, skip_con_x], axis=1)
        x = self.conv2(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        x = self.conv3(x)
        if self.use_bn:
            x = self.batchnorm(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.activation(x)
        return x


class FeatureMapBlock(nn.Module):
    """Feature Map block

    Final layer of U-Net which restores for the output channel dimensions to those of the input (or any other size)
    using a 1x1 convolution.

    Parameters
    ----------
    input_channels : :obj:`int`
        Number of input channels
    output_channels : :obj:`int`
        Number of output channels

    """
    def __init__(self, input_channels, output_channels):
        super(FeatureMapBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """UNet architecture

    UNet architecture composed of a series of contracting blocks followed by expanding blocks.

    Most UNet implementations available online hard-code a certain number of levels. Here,
    the number of levels for the contracting and expanding paths can be defined by the user and the
    UNet is built in such a way that the same code can be used for any number of levels without modification.

    Parameters
    ----------
    input_channels : :obj:`int`
        Number of input channels
    output_channels : :obj:`int`, optional
        Number of output channels
    hidden_channels : :obj:`int`, optional
        Number of hidden channels of first layer
    levels : :obj:`int`, optional
        Number of levels in encoding and deconding paths

    """
    def __init__(self, input_channels=1, output_channels=1, hidden_channels=64, levels=2):
        super(UNet, self).__init__()
        self.levels = levels
        self.upfeature = FeatureMapBlock(input_channels, hidden_channels)
        self.contract = []
        self.expand = []
        for level in range(levels):
            self.contract.append(ContractingBlock(hidden_channels * (2 ** level), use_dropout=False))
        for level in range(levels):
            self.expand.append(ExpandingBlock(hidden_channels * (2 ** (levels - level))))
        self.contracts = nn.Sequential(*self.contract)
        self.expands = nn.Sequential(*self.expand)
        self.downfeature = FeatureMapBlock(hidden_channels, output_channels)

    def forward(self, x):
        xenc = []
        x = self.upfeature(x)
        xenc.append(x)
        for level in range(self.levels):
            x = self.contract[level](x)
            xenc.append(x)
        for level in range(self.levels):
            x = self.expand[level](x, xenc[self.levels - level - 1])
        xn = self.downfeature(x)
        return xn