import torch
import torch.nn as nn
import os

from model.base_net import BaseNet
from utils import init_loader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# Check if cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Reproducibility setting
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'


def train(model, train_loader, test_loader, epoch, lr=0.001, weight_decay=0.001):
    """Trains and saves a model

    After each epoch, the model's accuracy on the train set and test set will be evaluated.
    After all epoch, the model's precision, recall and f1 values (for each class) on test set will be evaluated.
    :param model: Model to be trained
    :param train_loader: Data loader for the train set
    :param test_loader: Data loader for the test set
    :param epoch: Number of training epochs
    :param lr: Learning rate
    :param weight_decay: Weight decay
    :return: The trained model
    """
    # Init an optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Init the loss function
    criterion = nn.CrossEntropyLoss()

    batch = len(train_loader)

    # Iterate the training
    for e in range(epoch):
        # Lists for storing all ground truth labels and predictions for later accuracy evaluation
        gt_lst = []
        pred_lst = []

        model.train()
        # Train with each batch
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Refresh the gradients
            optimizer.zero_grad()

            # Forward, backward and optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Record the ground truth labels and predictions
            gt_lst += labels.tolist()
            _, preds = torch.max(outputs, 1)
            pred_lst += preds.tolist()

            # print(f'epoch {e + 1}/{epoch}, batch {i + 1}/{batch}, loss: {loss:.4f}')

        # Evaluates the model's accuracy on the train set and test set in this epoch
        train_accuracy = accuracy_score(gt_lst, pred_lst)
        test_accuracy = eval_accuracy(model, test_loader)
        print(f'epoch {e + 1}/{epoch}, train_accuracy: {train_accuracy}, test_accuracy: {test_accuracy}')

    return model


def eval_accuracy(model, test_loader):
    """Evaluates a model's accuracy on test set

    :param model: The model to be evaluated
    :param test_loader: Data loader for the test set
    :return: The model's accuracy on the test set
    """
    # Lists for storing all ground truth labels and predictions for later accuracy evaluation
    gt_lst = []
    pred_lst = []

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward
            outputs = model(inputs)

            gt_lst += labels.tolist()
            _, preds = torch.max(outputs, 1)
            pred_lst += preds.tolist()

        test_accuracy = accuracy_score(gt_lst, pred_lst)
        return test_accuracy


def eval_other_metric(model, test_loader):
    """Evaluates a model's precision, recall and f1 values (for each class) on test set

    :param model:
    :param test_loader:
    :return:
    """
    pass


if __name__ == '__main__':
    batch_size = 64
    train_loader, test_loader = init_loader(64)

    model = BaseNet().to(device)

    epoch = 10

    model = train(model, train_loader, test_loader, epoch)
