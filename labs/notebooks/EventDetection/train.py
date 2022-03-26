import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from dataset import plotting


def train(model, criterion, optimizer, data_loader, device='cpu'):
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
    device : :obj:`str`, optional
        Device

    Returns
    -------
    loss : :obj:`float`
        Loss over entire dataset
    accuracy : :obj:`float`
        Accuracy over entire dataset

    """
    model.train()
    accuracy = 0
    loss = 0
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        yprob = model(X)
        ls = criterion(yprob.view(-1), y.view(-1))
        ls.backward()
        optimizer.step()
        with torch.no_grad(): # use no_grad to avoid making the computational graph...
            y_pred = np.where(nn.Sigmoid()(yprob.detach()).cpu().numpy() > 0.5, 1, 0).astype(np.float32)
        loss += ls.item()
        accuracy += accuracy_score(y.cpu().numpy().ravel(), y_pred.ravel())
    loss /= len(data_loader)
    accuracy /= len(data_loader)
    return loss, accuracy


def evaluate(model, criterion, data_loader, device='cpu'):
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
    device : :obj:`str`, optional
        Device

    Returns
    -------
    loss : :obj:`float`
        Loss over entire dataset
    accuracy : :obj:`float`
        Accuracy over entire dataset

    """
    model.eval()
    accuracy = 0
    loss = 0
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        with torch.no_grad(): # use no_grad to avoid making the computational graph...
            yprob = model(X)
            ls = criterion(yprob.view(-1), y.view(-1))
            y_pred = np.where(nn.Sigmoid()(yprob.detach()).cpu().numpy() > 0.5, 1, 0).astype(np.float32)
        loss += ls.item()
        accuracy += accuracy_score(y.cpu().numpy().ravel(), y_pred.ravel())
    loss /= len(data_loader)
    accuracy /= len(data_loader)
    return loss, accuracy


def predict(model, X, y, label, device='cpu', dt=0.002, nplot=5, report=False):
    """Prediction step

    Perform a prediction over a batch of input samples

    Parameters
    ----------
    model : :obj:`torch.nn.Module`
        Model
    X : :obj:`torch.tensor`
        Inputs
    y : :obj:`torch.tensor`
        Masks
    label : :obj:`str`
        Label to use in plotting
    device : :obj:`str`, optional
        Device

    """
    model.eval()
    X = X.to(device)

    with torch.no_grad():  # use no_grad to avoid making the computational graph...
        yprob = nn.Sigmoid()(model(X))
    y_pred = np.where(yprob.cpu().numpy() > 0.5, 1, 0)

    if report:
        print(classification_report(y.ravel(), y_pred.ravel()))

    plotting(X.cpu().detach().numpy().squeeze(),
             y, X.cpu().detach().numpy().squeeze(),
             y_pred, y2prob=yprob.cpu().numpy(),
             title1='True', title2=label, dt=dt, nplot=nplot)


def training(model, loss, optim, nepochs, train_loader, test_loader,
             device='cpu', modeldir=None, modelname=''):
    """Training

    Perform full training cycle

    Parameters
    ----------
    model : :obj:`torch.nn.Module`
        Model
    loss : :obj:`torch.nn.modules.loss`
        Loss function
    optim : :obj:`torch.optim`
        Optimizer
    nepochs : :obj:`int`, optional
        Number of epochs
    train_loader : :obj:`torch.utils.data.dataloader.DataLoader`
        Training dataloader
    test_loader : :obj:`torch.utils.data.dataloader.DataLoader`
        Testing dataloader
    device : :obj:`str`, optional
        Device
    modeldir : :obj:`str`, optional
        Directory where to save model (if ``None``, do not save model)

    """
    iepoch_best = 0
    train_loss_history = np.zeros(nepochs)
    valid_loss_history = np.zeros(nepochs)
    train_acc_history = np.zeros(nepochs)
    valid_acc_history = np.zeros(nepochs)
    for i in range(nepochs):
        train_loss, train_accuracy = train(model, loss, optim,
                                           train_loader, device=device)
        valid_loss, valid_accuracy = evaluate(model, loss,
                                              test_loader, device=device)
        train_loss_history[i] = train_loss
        valid_loss_history[i] = valid_loss
        train_acc_history[i] = train_accuracy
        valid_acc_history[i] = valid_accuracy
        if modeldir is not None:
            if i == 0 or valid_accuracy > np.max(valid_acc_history[:i]):
                iepoch_best = i
                torch.save(model.state_dict(), os.path.join(modeldir, 'models', modelname+'.pt'))
        if i % 10 == 0:
            print(f'Epoch {i}, Training Loss {train_loss:.3f}, Training Accuracy {train_accuracy:.3f}, Test Loss {valid_loss:.3f}, Test Accuracy {valid_accuracy:.3f}')
    return train_loss_history, valid_loss_history, train_acc_history, valid_acc_history, iepoch_best