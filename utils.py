from sklearn.decomposition import PCA

import torch
import torchvision
import torchvision.transforms as transforms


def init_loader(batch_size=16, full=False):
    """Inits train loader and test loader for CIFAR-10

    The first param controls the batch size of the loader when the second param is False.
    If the second is True, the batch size will be equal to the size of the whole dataset.
    By default, it returns loaders with a batch size of 16
    :param batch_size: Batch size for the loaders
    :type batch_size: int
    :param full: If the whole dataset will be returned as one batch
    :type full: bool

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


def apply_pca(dim, train_x, test_x):
    """Applies the PCA trained on X of train data on X of train and test data

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


def cross_val_pca(model, fold):
    """Cross validation for the model that needs to use PCA processed data

    :return: Precision, recall, f1 values (for each class) and accuracy of the model
    """


if __name__ == '__main__':
    train_loader, test_loader = init_loader()
