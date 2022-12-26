import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# Fix the seed so that the data from each run stays the same
seed = 10
torch.manual_seed(seed)


def get_root_dir():
    """Gets the root path of the current project

    :return: The root path of the current project
    """
    root_dir = os.path.dirname(os.path.abspath(__file__))
    return root_dir


def get_cls_name():
    """Get the class names

    :return: The list of class names of CIFAR-10
    """
    return ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


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

    # Use first 5000 pieces of data as train set
    train_set = torch.utils.data.Subset(train_set, list(range(5000)))

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
    # Get a generator from the loader
    data = next(iter(loader))

    # If size is not -1, slice the data to the specified size
    if size != -1:
        data = [data[0][:size], data[1][:size]]

    return data


def plot_multi_line(hyper_param_name, metric_name, metric_lst, hyper_param_lst, test):
    """Plots and saves a line chart with multiple lines, where x-axis takes the names of classes and each metric value
    in metric_lst corresponds to a line

    Save the plot in a file whose name follows the format: param_name-param_str-metric_name-phase.png.
    For example: dim-3072_2000_1000-precision-test.png
    :param hyper_param_name: The name of the hyperparameter
    :param metric_name: Name of the metric
    :param metric_lst: List of the metric value
    :param hyper_param_lst: Hyperparameters corresponding to accuracy values
    :param test: If this plot is generated during test or validation
    """
    # Clear the previous content
    plt.clf()
    plt.cla()

    # Get the class names
    cls_lst = get_cls_name()
    x = range(len(cls_lst))

    # Avoid converting a float such as 0.3 to str '0.30000000001'
    str_hyper_param_lst = [str(x) for x in hyper_param_lst] if isinstance(hyper_param_lst[0], int) else [f'{x:.2f}' for x in hyper_param_lst]

    # Plot each metric value
    for idx, metric in enumerate(metric_lst):
        plt.plot(x, metric, marker='o', label=str_hyper_param_lst[idx])

    # Show the legend
    plt.legend()
    # Show class names on the x-axis
    plt.xticks(x, cls_lst, rotation=45)
    plt.ylabel(metric_name)

    # Save the plot
    phase = 'test' if test else 'val'
    hyper_param_str = '_'.join(str_hyper_param_lst)
    root_dir = get_root_dir()
    if not(os.path.exists(f'{root_dir}/result')):
        os.makedirs(f'{root_dir}/result')
    plt.savefig(f'{root_dir}/result/{hyper_param_name}-{hyper_param_str}-{metric_name}-{phase}.png')


def plot_single_line(hyper_param_name, hyper_param_lst, accuracy_lst, test):
    """Plots and save a line chart with a single line, where x-axis takes the hyperparameters

    The naming rule of the file is the same as `plot_multi_line`.
    :param hyper_param_name: The name of the hyperparameter
    :param hyper_param_lst: Hyperparameters corresponding to accuracy values
    :param accuracy_lst: List of the accuracy value
    :param test: If this plot is generated during test or validation
    """
    # Clear the previous content
    plt.clf()
    plt.cla()

    # Plot the accuracy
    plt.plot(hyper_param_lst, accuracy_lst, marker='o')

    plt.xlabel(hyper_param_name)
    plt.ylabel('accuracy')

    # Save the plot
    phase = 'test' if test else 'val'
    # Avoid converting a float such as 0.3 to str '0.30000000001'
    str_hyper_param_lst = [str(x) for x in hyper_param_lst] if isinstance(hyper_param_lst[0], int) else [f'{x:.2f}' for x in hyper_param_lst]
    hyper_param_str = '_'.join(str_hyper_param_lst)
    root_dir = get_root_dir()
    plt.savefig(f'{root_dir}/result/{hyper_param_name}-{hyper_param_str}-accuracy-{phase}.png')


def plot_result(hyper_param_name, hyper_param_lst, precision_lst, recall_lst, f1_lst, accuracy_lst, test):
    """Plots and saves the result

    :param hyper_param_name: The name of the hyperparameter
    :param hyper_param_lst: List of the hyperparameters
    :param precision_lst: List of the precision value
    :param recall_lst: List of the recall value
    :param f1_lst: List of the f1 value
    :param accuracy_lst: List of the accuracy value
    :param test: If this plot is generated during test or validation
    """
    plot_multi_line(hyper_param_name, 'precision', precision_lst, hyper_param_lst, test)
    plot_multi_line(hyper_param_name, 'recall', recall_lst, hyper_param_lst, test)
    plot_multi_line(hyper_param_name, 'f1', f1_lst, hyper_param_lst, test)
    plot_single_line(hyper_param_name, hyper_param_lst, accuracy_lst, test)


if __name__ == '__main__':
    train_loader, test_loader = init_loader(full=True)
    train_data = get_data_once(train_loader, 5000)