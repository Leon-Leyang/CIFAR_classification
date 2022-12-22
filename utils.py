from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import math
import torch
import torchvision
import torchvision.transforms as transforms


def init_loader(batch_size=16, full=False):
    """Inits train loader and test loader for CIFAR-10

    The first param controls the batch size of the loader when the second param is False.
    If the second is True, the batch size will be equal to the size of the whole dataset.
    By default, it returns loaders with a batch size of 16
    :param batch_size: Batch size for the loaders
    :param full: If the whole dataset will be returned as one batch

    :return: train loader, test loader
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)

    if not full:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                   shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                                  shuffle=False, num_workers=2)
    else:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=len(train_set),
                                                   shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set),
                                                  shuffle=False, num_workers=2)

    return train_loader, test_loader


def get_subset(loader, size):
    """Gets a subset from the loader of the specified size

    If the size is -1, the whole data will be returned.
    :param loader: Data loader
    :param size: Size of the subset
    :return: The subset of the specified size
    """
    # Fix the seed so that the data from each run stays the same
    seed = 10
    torch.manual_seed(seed)

    # Get a generator from the loader
    data = next(iter(loader))

    # Get the subset of the data
    sub_data = [data[0][:size], data[1][:size]]

    return sub_data


def apply_pca(dim, train_x, test_x):
    """Trains a PCA on X of train data and applies it on X of train and test data

    :param dim: Dimension of the data after PCA
    :param train_x: X of train data of shape B*C*H*W
    :param test_x: X of test data of shape B*C*H*W
    :return: X of train and test data after PCA
    """
    # Flatten the data of shape B*C*H*W to B*(C*H*W)
    f_train_x = torch.flatten(train_x, start_dim=1)
    f_test_x = torch.flatten(test_x, start_dim=1)

    # Train a PCA with train data
    pca = PCA(n_components=dim)
    pca.fit(f_train_x)

    # Transform the data
    new_train_x = pca.transform(f_train_x)
    new_test_x = pca.transform(f_test_x)

    return new_train_x, new_test_x


def cross_val_pca(dim, fold, full_train_data):
    """Cross validation for a SVM trained on data whose dimension is reduced to a specific value with PCA

    :return: Average precision, recall, f1 values (for each class) and accuracy of the SVM
    """
    # Lists that store the values of metrics in each round
    precision_lst = []
    recall_lst = []
    f1_lst = []
    accuracy_lst = []

    # Init a SVM
    clf = svm.SVC(kernel='linear')

    # Calculate the number of data per fold
    num_total = full_train_data[0].shape[0]
    num_per_fold = math.ceil(num_total / fold)

    # Cross validation
    for idx in range(fold):
        # Get the train data and val data for this round
        # If the selected val set is not the last fold
        if idx != fold - 1:
            val_x = full_train_data[0][idx * num_per_fold:(idx + 1) * num_per_fold]
            val_y = full_train_data[1][idx * num_per_fold:(idx + 1) * num_per_fold]
            train_x = torch.cat((full_train_data[0][:idx * num_per_fold], full_train_data[0][(idx + 1) * num_per_fold:]), 0)
            train_y = torch.cat((full_train_data[1][:idx * num_per_fold], full_train_data[1][(idx + 1) * num_per_fold:]), 0)
        # If the selected val set is the last fold
        else:
            val_x = full_train_data[0][idx * num_per_fold:]
            val_y = full_train_data[1][idx * num_per_fold:]
            train_x = full_train_data[0][:idx * num_per_fold]
            train_y = full_train_data[1][:idx * num_per_fold]

        # Process train_x and val_x with PCA if dim is not equal to -1
        if dim != -1:
            train_x, val_x = apply_pca(dim, train_x, val_x)
        # print(f'{train_x.shape}, {train_y.shape}, {val_x.shape}, {val_y.shape}')

        # Train the SVM
        clf.fit(train_x, train_y)

        # Eval the SVM
        pred = clf.predict(val_x)
        precision = precision_score(val_y, pred, average=None)
        recall = recall_score(val_y, pred, average=None)
        f1 = f1_score(val_y, pred, average=None)
        accuracy = accuracy_score(val_y, pred)
        print(f'round {idx + 1}\nprecision: {precision}\nrecall: {recall}\nf1: {f1}\naccuracy: {accuracy}\n')


if __name__ == '__main__':
    train_loader, test_loader = init_loader(full=True)

    train_data = get_subset(train_loader, 5000)

    cross_val_pca(1000, 5, train_data)