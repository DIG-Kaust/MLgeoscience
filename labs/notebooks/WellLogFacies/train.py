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
        Loss over entire training dataset
    accuracy : :obj:`float`
        Accuracy over entire training dataset

    """
    model.train()
    loss = 0.
    accuracy = 0.
    for X, y in data_loader:
        optimizer.zero_grad()
        yprob = model(X)
        ls = criterion(yprob, y)
        ls.backward()
        optimizer.step()
        with torch.no_grad():
            y_pred = np.argmax(nn.Softmax(dim=1)(yprob).detach().numpy(), axis=1)
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
        Evaluation dataloader

    Returns
    -------
    loss : :obj:`float`
        Loss over entire evaluation dataset
    accuracy : :obj:`float`
        Accuracy over entire evaluation dataset

    """
    model.eval()
    loss = 0.
    accuracy = 0.
    for X, y in data_loader:
        with torch.no_grad(): # use no_grad to avoid making the computational graph...
            yprob = model(X)
            ls = criterion(yprob, y)
            y_pred = np.argmax(nn.Softmax(dim=1)(yprob).detach().numpy(), axis=1)
        loss += ls.item()
        accuracy += accuracy_score(y, y_pred)
    loss /= len(data_loader)
    accuracy /= len(data_loader)
    return loss, accuracy


def classification(model, train_loader, valid_loader, epochs=1000):
    """Classifier

    Perform binary classification

    Parameters
    ----------
    model : :obj:`torch.nn.Module`
        Model
    train_loader : :obj:`torch.utils.data.dataloader.DataLoader`
        Training dataloader
    valid_loader : :obj:`torch.utils.data.dataloader.DataLoader`
        Validation dataloader
    epochs : :obj:`int`, optional
        Number of epochs

    """
    bce_loss = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

    train_loss_history = np.zeros(epochs)
    valid_loss_history = np.zeros(epochs)
    train_acc_history = np.zeros(epochs)
    valid_acc_history = np.zeros(epochs)
    for i in range(epochs):
        train_loss, train_acc = train(model, bce_loss, optim, train_loader)
        valid_loss, valid_acc = evaluate(model, bce_loss, valid_loader)
        train_loss_history[i] = train_loss
        valid_loss_history[i] = valid_loss
        train_acc_history[i] = train_acc
        valid_acc_history[i] = valid_acc
        if i % 10 == 0:
            print(f'Epoch {i}, Training Loss {train_loss:.2f}, Training Accuracy {train_acc:.2f}, Validation Loss {valid_loss:.2f}, Test Accuracy {valid_acc:.2f}')
            
    return train_loss_history, valid_loss_history, train_acc_history, valid_acc_history
