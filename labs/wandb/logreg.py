#!/usr/bin/env python3
"""
Logistic regression with Wandb logging

"""
import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb

from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary

def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out
    any randomness from cuda kernels
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    return True

def make_train_test(train_size, test_size, noise=0.05):
    """
    Makes a two-moon train-test dataset
    """
    X_train, y_train = make_moons(n_samples=train_size, noise=noise)
    y_train = y_train.reshape(train_size, 1)
    X_train = X_train.reshape(train_size, 2)

    X_test, y_test = make_moons(n_samples=test_size, noise=0.1)
    y_test = y_test.reshape(test_size, 1)
    return X_train, y_train, X_test, y_test

def train(model, criterion, optimizer, data_loader):
    model.train()
    accuracy = 0
    for X, y in data_loader:
        optimizer.zero_grad()
        yprob = model(X)
        loss = criterion(yprob, y)
        loss.backward()
        optimizer.step()
        y_pred = np.where(yprob[:, 0].detach().numpy() > 0.5, 1, 0)
        accuracy += accuracy_score(y, y_pred)
    accuracy /= len(data_loader)
    # store train accuracy in wandb logger
    wandb.log({"train_accuracy": accuracy})
    return loss, accuracy

def evaluate(model, criterion, data_loader):
    model.eval()
    accuracy = 0
    for X, y in data_loader:
        with torch.no_grad(): # use no_grad to avoid making the computational graph...
            yprob = model(X)
            loss = criterion(yprob, y)
        y_pred = np.where(yprob[:, 0].numpy() > 0.5, 1, 0)
        accuracy = accuracy_score(y, y_pred)
    accuracy /= len(data_loader)
    # store valid accuracy in wandb logger
    wandb.log({"valid_accuracy": accuracy})
    return loss, accuracy

class SingleHiddenLayerNetwork(nn.Module):
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


def main(parser):

    ######### INPUT PARAMS #########
    parser.add_argument('project', type=str, help='Project name')
    parser.add_argument('-u', '--hidden', type=int,  default=10, help='Size of hidden layer')
    parser.add_argument('-l', '--learningrate', type=float,  default=0.1, help='Learning Rate')
    parser.add_argument('-e', '--nepochs', type=int,  default=100, help='Number of Epochs')

    args = parser.parse_args()

    # Define wandb project to use
    wandb.init(project=args.project)

    # Crete configure object to which we can attach the parameters to log
    config = wandb.config

    # Create training data
    set_seed(42)

    train_size = 1000 # Size of training data
    test_size = 200 # Size of test data

    X_train, y_train, X_test, y_test = make_train_test(train_size, test_size, noise=0.2)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='Set2')
    ax[0].set_title('Train data')
    ax[1].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='Set2')
    ax[1].set_title('Test data')

    # Define Train Set
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()

    train_dataset = TensorDataset(X_train, y_train)

    # Define Test Set
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()

    test_dataset = TensorDataset(X_test, y_test)

    # Use Pytorch's functionality to load data in batches. Here we use full-batch training again.
    train_loader = DataLoader(train_dataset, batch_size=X_train.size(0), shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=X_test.size(0), shuffle=False)

    # Define experiment parameters to be logged
    input = 2
    output = 1
    config.hidden = args.hidden
    config.learning_rate = args.learningrate
    config.nepochs = args.nepochs

    # Train
    network = SingleHiddenLayerNetwork(input, config.hidden, output)
    bce_loss = nn.BCELoss()
    optim = torch.optim.SGD(network.parameters(), lr=config.learning_rate)

    for i in range(config.nepochs):
        train_loss, train_accuracy = train(network, bce_loss, optim, train_loader)
        test_loss, test_accuracy = evaluate(network, bce_loss, test_loader)

        if i % 100 == 0:
            print(
                f'Epoch {i}, Training Loss {train_loss.item():.2f}, '
                f'Training Accuracy {train_accuracy:.2f}, '
                f'Test Loss {test_loss.item():.2f}, '
                f'Test Accuracy {test_accuracy:.2f}')

    network.eval()
    with torch.no_grad():
        a_train = network(X_train)
        a_test = network(X_test)
    print("Test set accuracy: ", accuracy_score(y_test, np.where(a_test[:, 0].numpy()>0.5, 1, 0)))

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].scatter(X_train[:, 0], X_train[:, 1], c=np.where(a_train[:, 0].numpy()>0.5, 1, 0), cmap='Set2')
    ax[0].set_title('Train data')
    ax[1].scatter(X_test[:, 0], X_test[:, 1], c=np.where(a_test[:, 0].numpy()>0.5, 1, 0), cmap='Set2')
    ax[0].set_title('Test data')

    plt.show()



if __name__ == "__main__":

    description = 'Logistic Regression with wandb'
    main(argparse.ArgumentParser(description=description))

