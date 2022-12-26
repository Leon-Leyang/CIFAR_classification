import torch
import torch.nn as nn

from model.base_net import BaseNet
from utils import init_loader


def train(model, train_loader, epoch, lr, weight_decay=0.001):
    """Trains and saves a model

    :param model: Model to be trained
    :param train_loader:Data loader for the train data
    :param epoch: Number of training epochs
    :param lr: Learning rate
    :param weight_decay: Weight decay
    :return: The trained model
    """
    # Init an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Init the loss function
    criterion = nn.CrossEntropyLoss()

    # Iterate the training
    for e in range(epoch):
        model.train()

        # Train with each batch
        for i, data in enumerate(train_loader):
            inputs, labels = data

            # Refresh the gradients
            optimizer.zero_grad()

            # Forward, backward and optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print(f'epoch {e + 1}/{epoch}')



def test(model, test_loader):
    pass


if __name__ == '__main__':
    batch_size = 64
    train_loader, test_loader = init_loader(64)

    model = BaseNet()

    epoch = 10

    model = train(model, train_loader, epoch)

    test(model, test_loader)
