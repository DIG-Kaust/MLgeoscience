import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score

from utils import show_tensor_images


def train(model, criterion, optimizer, data_loader, device='cpu', plotflag=False):
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
    plotflag : :obj:`bool`, optional
        Display intermediate results

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
    for dl in tqdm(data_loader):
        X, y = dl['image'], dl['mask']
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        yprob = model(X)
        ls = criterion(yprob.view(-1), y.view(-1))
        ls.backward()
        optimizer.step()
        with torch.no_grad():
            yprob = nn.Sigmoid()(yprob)
            ypred = (yprob.detach().cpu().numpy() > 0.5).astype(float)
        loss += ls.item()
        accuracy += accuracy_score(y.cpu().numpy().ravel(), ypred.ravel())
    loss /= len(data_loader)
    accuracy /= len(data_loader)

    if plotflag:
        fig, axs = plt.subplots(2, 2, figsize=(16, 9))
        show_tensor_images(X, ax=axs[0][0], num_images=15, vmin=-1, vmax=1)
        axs[0][0].set_title("Images")
        axs[0][0].axis('tight')
        show_tensor_images(y, ax=axs[0][1], num_images=15, vmin=0, vmax=1)
        axs[0][1].set_title("Mask")
        axs[0][1].axis('tight')
        show_tensor_images(yprob, ax=axs[1][0], num_images=15, vmin=0, vmax=1)
        axs[1][0].set_title("Reconstructed Mask (Prob.)")
        axs[1][0].axis('tight')
        show_tensor_images(torch.from_numpy(ypred), ax=axs[1][1], num_images=15, vmin=0, vmax=1)
        axs[1][1].set_title("Reconstructed Mask")
        axs[1][1].axis('tight')
        plt.show()
    return loss, accuracy


def evaluate(model, criterion, data_loader, device='cpu', plotflag=False):
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
    device : :obj:`str`, optional
        Device
    plotflag : :obj:`bool`, optional
        Display intermediate results

    Returns
    -------
    loss : :obj:`float`
        Loss over entire dataset
    accuracy : :obj:`float`
        Accuracy over entire dataset

    """
    model.train()  # not eval because https://github.com/facebookresearch/SparseConvNet/issues/166
    accuracy = 0
    loss = 0
    for dl in data_loader:
        X, y = dl['image'], dl['mask']
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            yprob = model(X)
            ls = criterion(yprob.view(-1), y.view(-1))
            yprob = nn.Sigmoid()(yprob)
            ypred = (yprob.detach().cpu().numpy() > 0.5).astype(float)
        loss += ls.item()
        accuracy += accuracy_score(y.cpu().numpy().ravel(), ypred.ravel())
    loss /= len(data_loader)
    accuracy /= len(data_loader)

    if plotflag:
        fig, axs = plt.subplots(2, 2, figsize=(16, 9))
        show_tensor_images(X, ax=axs[0][0], num_images=15, vmin=-1, vmax=1)
        axs[0][0].set_title("Images")
        axs[0][0].axis('tight')
        show_tensor_images(y, ax=axs[0][1], num_images=15, vmin=0, vmax=1)
        axs[0][1].set_title("Mask")
        axs[0][1].axis('tight')
        show_tensor_images(yprob, ax=axs[1][0], num_images=15, vmin=0, vmax=1)
        axs[1][0].set_title("Reconstructed Mask (Prob.)")
        axs[1][0].axis('tight')
        show_tensor_images(torch.from_numpy(ypred), ax=axs[1][1], num_images=15, vmin=0, vmax=1)
        axs[1][1].set_title("Reconstructed Mask")
        axs[1][1].axis('tight')
        plt.show()
    return loss, accuracy


def predict(model, X, y, device='cpu'):
    """Prediction step

    Perform a prediction over a batch of input samples

    Parameters
    ----------
    model : :obj:`torch.nn.Module`
        Model
    X : :obj:`torch.tensor`
        Inputs
    X : :obj:`torch.tensor`
        Masks
    device : :obj:`str`, optional
        Device

    """
    model.train()  # not eval because https://github.com/facebookresearch/SparseConvNet/issues/166
    X, y = X.to(device), y.to(device)
    # or create statistics (https://discuss.pytorch.org/t/model-eval-gives-incorrect-loss-for-model-with-batchnorm-layers/7561/2)
    # network(X)
    # model.eval()
    yprob = model(X)
    with torch.no_grad():
        yprob = nn.Sigmoid()(yprob)
        y_pred = (yprob.detach().cpu().numpy() > 0.5).astype(float)

    fig, axs = plt.subplots(2, 2, figsize=(16, 9))
    show_tensor_images(X, ax=axs[0][0], num_images=15, vmin=-1, vmax=1, cbar=False)
    axs[0][0].set_title("Images")
    axs[0][0].axis('tight')
    show_tensor_images(y, ax=axs[0][1], num_images=15, vmin=0, vmax=1, cbar=False)
    axs[0][1].set_title("Mask")
    axs[0][1].axis('tight')
    show_tensor_images(yprob, ax=axs[1][0], num_images=15, vmin=0, vmax=1, cbar=False)
    axs[1][0].set_title("Reconstructed Mask (Prob.)")
    axs[1][0].axis('tight')
    show_tensor_images(torch.from_numpy(y_pred), ax=axs[1][1], num_images=15, vmin=0, vmax=1, cbar=False)
    axs[1][1].set_title("Reconstructed Mask")
    axs[1][1].axis('tight')
    plt.tight_layout()