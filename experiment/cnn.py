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
        test_accuracy = eval(model, test_loader, ['accuracy'])[0]
        print(f'epoch {e + 1}/{epoch}, train_accuracy: {train_accuracy}, test_accuracy: {test_accuracy}')

    print('\nFinish training\n')

    # Evaluates other metrics of the final model
    precision, recall, f1 = eval(model, test_loader, ['precision', 'recall', 'f1'])
    print(f'precision: {precision}\nrecall: {recall}\nf1: {f1}\naccuracy: {test_accuracy}\n')


def eval(model, test_loader, metric_lst):
    """Evaluates a model on test set

    :param model: The model to be evaluated
    :param test_loader: Data loader for the test set
    :return: A list to store the return result
    """
    # Store the mapping from metric name to function
    name_2_func = {'precision': precision_score, 'recall': recall_score, 'f1': f1_score, 'accuracy': accuracy_score}

    # Return list
    ret_lst = []

    # Lists for storing all ground truth labels and predictions for later evaluation
    gt_lst = []
    pred_lst = []

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward
            outputs = model(inputs)

            # Record the ground truth labels and predictions
            gt_lst += labels.tolist()
            _, preds = torch.max(outputs, 1)
            pred_lst += preds.tolist()

    # Iterate through each metric
    for name in metric_lst:
        func = name_2_func[name]
        if name == 'accuracy':
            value = func(gt_lst, pred_lst)
        else:
            value = func(gt_lst, pred_lst, average=None)

        ret_lst.append(value)
    return ret_lst


if __name__ == '__main__':
    batch_size = 64
    train_loader, test_loader = init_loader(64)
    model = BaseNet().to(device)
    epoch = 5

    train(model, train_loader, test_loader, epoch)
