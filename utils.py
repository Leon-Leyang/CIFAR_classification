import torch
import torchvision
import torchvision.transforms as transforms


def init_loader(batch_size=16):
    """Inits train loader and test loader for CIFAR-10

    Default batch size is 16.
    :param batch_size: Batch size for the loaders
    :type batch_size: int

    :return: train loader, test loader
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=2)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

    return train_loader, test_loader


if __name__ == '__main__':
    train_loader, test_loader = init_loader()
