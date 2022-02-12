import torch
from sklearn.datasets import make_moons
from torch.utils.data import TensorDataset, DataLoader


def make_train_test(train_size, test_size, noise=0.05):
    """Makes a two-moon train-test dataset

    Parameters
    ----------
    train_size : :obj:`int`
        Number of training samples
    test_size : :obj:`int`
        Number of test samples
    noise : :obj:`float`, optional
        Standard deviation of noise

    Returns
    -------
    X_train : :obj:`torch.Tensor`
        Training input samples
    y_train : :obj:`torch.Tensor`
        Training output samples
    X_test : :obj:`torch.Tensor`
        Testing input samples
    y_test : :obj:`torch.Tensor`
        Testing output samples
    train_loader : :obj:`torch.utils.data.dataloader.DataLoader`
        Training dataloader
    test_loader : :obj:`torch.utils.data.dataloader.DataLoader`
        Testing dataloader

    """
    # Create dataset
    X_train, y_train = make_moons(n_samples=train_size, noise=noise)
    y_train = y_train.reshape(train_size, 1)
    X_train = X_train.reshape(train_size, 2)

    X_test, y_test = make_moons(n_samples=test_size, noise=0.1)
    y_test = y_test.reshape(test_size, 1)

    # Convert Train Set to Torch
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    train_dataset = TensorDataset(X_train, y_train)

    # Define Test Set to Torch
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()
    test_dataset = TensorDataset(X_test, y_test)

    # Use Pytorch's functionality to load data in batches.
    train_loader = DataLoader(train_dataset, batch_size=X_train.size(0),
                              shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=X_test.size(0),
                             shuffle=False)

    return X_train, y_train, X_test, y_test, train_loader, test_loader