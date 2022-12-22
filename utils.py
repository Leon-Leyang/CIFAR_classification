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


if __name__ == '__main__':
    train_loader, test_loader = init_loader(full=True)

    train_data = get_subset(train_loader, 5000)
