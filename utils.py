import os
import torch
import torchvision
import torchvision.transforms as transforms


def get_root_dir():
    """Gets the root path of the current project

    :return: The root path of the current project
    """
    root_dir = os.path.dirname(os.path.abspath(__file__))
    return root_dir


def init_loader(batch_size=16, full=False):
    """Inits train loader and test loader for CIFAR-10

    The first param controls the batch size of the loader when the second param is False.
    If the second is True, the batch size will be equal to the size of the whole dataset.
    By default, it returns loaders with a batch size of 16
    :param batch_size: Batch size for the loaders
    :param full: If the whole dataset will be returned as one batch

    :return: train loader, test loader
    """
    # Get the root directory of the project
    root_dir = get_root_dir()

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root=f'{root_dir}/data', train=True,
                                             download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root=f'{root_dir}/data', train=False,
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


def get_data_once(loader, size):
    """Gets a number of data from the loader at once

    If the size is -1, all data in the data loader will be returned.
    :param loader: Data loader
    :param size: Size of the returned data
    :return: The data of the specified size
    """
    # Fix the seed so that the data from each run stays the same
    seed = 10
    torch.manual_seed(seed)

    # Get a generator from the loader
    data = next(iter(loader))

    # If size is not -1, slice the data to the specified size
    if size != -1:
        data = [data[0][:size], data[1][:size]]

    return data


if __name__ == '__main__':
    train_loader, test_loader = init_loader(full=True)
    train_data = get_data_once(train_loader, 5000)