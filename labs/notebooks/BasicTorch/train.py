import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score


def train(model, criterion, optimizer, data_loader):
    """Training step

    Perform a training step over the entire training data (1 epoch of training)

    Parameters
    ----------
    model : :obj:`torch.nn.Module`
        Model
    criterion : :obj:`torch.nn.modules.loss`
        Loss function
    optimizer : :obj:`torch.optim`
        Optimizer
    data_loader : :obj:`torch.utils.data.dataloader.DataLoader`
        Training dataloader

    Returns
    -------
    loss : :obj:`float`
        Loss over entire dataset
    accuracy : :obj:`float`
        Accuracy over entire dataset

    """
    model.train()
    loss = 0
    accuracy = 0
    for X, y in data_loader:
        optimizer.zero_grad()
        yprob = model(X)
        ls = criterion(yprob, y)
        ls.backward()
        optimizer.step()
        y_pred = np.where(yprob[:, 0].detach().numpy() > 0.5, 1, 0)
        loss += ls.item()
        accuracy += accuracy_score(y, y_pred)
    loss /= len(data_loader)
    accuracy /= len(data_loader)
    return loss, accuracy


def evaluate(model, criterion, data_loader):
    """Evaluation step

    Perform an evaluation step over the entire training data

    Parameters
    ----------
    model : :obj:`torch.nn.Module`
        Model
    criterion : :obj:`torch.nn.modules.loss`
        Loss function
    data_loader : :obj:`torch.utils.data.dataloader.DataLoader`
        Training dataloader

    Returns
    -------
    loss : :obj:`float`
        Loss over entire dataset
    accuracy : :obj:`float`
        Accuracy over entire dataset

    """
    model.eval()
    loss = 0
    accuracy = 0
    for X, y in data_loader:
        with torch.no_grad(): # use no_grad to avoid making the computational graph...
            yprob = model(X)
            ls = criterion(yprob, y)
        y_pred = np.where(yprob[:, 0].numpy() > 0.5, 1, 0)
        loss += ls.item()
        accuracy += accuracy_score(y, y_pred)
    loss /= len(data_loader)
    accuracy /= len(data_loader)
    return loss, accuracy


def classification(model, train_loader, test_loader, epochs=1000):
    """Classifier

    Perform binary classification

    Parameters
    ----------
    model : :obj:`torch.nn.Module`
        Model
    train_loader : :obj:`torch.utils.data.dataloader.DataLoader`
        Training dataloader
    test_loader : :obj:`torch.utils.data.dataloader.DataLoader`
        Testing dataloader
    epochs : :obj:`int`, optional
        Number of epochs

    """
    bce_loss = nn.BCELoss()
    optim = torch.optim.SGD(model.parameters(), lr=1)

    for i in range(epochs):
        train_loss, train_accuracy = train(model, bce_loss, optim, train_loader)
        test_loss, test_accuracy = evaluate(model, bce_loss, test_loader)

        if i % 100 == 0:
            print(f'Epoch {i}, Training Loss {train_loss:.2f}, Training Accuracy {train_accuracy:.2f}, Test Loss {test_loss:.2f}, Test Accuracy {test_accuracy:.2f}')