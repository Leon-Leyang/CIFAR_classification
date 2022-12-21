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


def apply_pca(dim, train_data, val_data, test_data):
    """Applys the PCA trained on train data on train, validation and test data

    :param dim: Dimension of the data after PCA
    :param train_data: Train data of shape B*C*H*W
    :param val_data: Validation data of shape B*C*H*W
    :param test_data: Test data of shape B*C*H*W
    :return: train, validation and test data after PCA
    """
    # Flatten the data of shape B*C*H*W to B*(C*H*W)
    f_train_data = torch.flatten(train_data, start_dim=1)
    f_val_data = torch.flatten(val_data, start_dim=1)
    f_test_data = torch.flatten(test_data, start_dim=1)

    # Train a PCA with train data
    pca = PCA(n_components=dim)
    pca.fit(f_train_data)

    # Transform the data
    new_train_data = pca.transform(f_train_data)
    new_val_data = pca.transform(f_val_data)
    new_test_data = pca.transform(f_test_data)

    return new_train_data, new_val_data, new_test_data


if __name__ == '__main__':
    train_loader, test_loader = init_loader()
