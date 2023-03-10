import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from matplotlib.pyplot import MaxNLocator


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


def plot_other_metric_svm(hyper_param_name, metric_name, metric_lst, hyper_param_lst, test):
    """Plots and saves a line chart for SVM experiment with multiple lines, where x-axis takes the names of classes and each metric value
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
    plt.savefig(f'{root_dir}/result/{hyper_param_name}-{hyper_param_str}-{metric_name}-{phase}.png')


def plot_accuracy_svm(hyper_param_name, hyper_param_lst, accuracy_lst, test):
    """Plots and save a line chart for accuracy of SVM experiment with a single line, where x-axis takes the hyperparameters

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


def plot_result_svm(hyper_param_name, hyper_param_lst, precision_lst, recall_lst, f1_lst, accuracy_lst, test):
    """Plots and saves the result for SVM experiment

    :param hyper_param_name: The name of the hyperparameter
    :param hyper_param_lst: List of the hyperparameters
    :param precision_lst: List of the precision value
    :param recall_lst: List of the recall value
    :param f1_lst: List of the f1 value
    :param accuracy_lst: List of the accuracy value
    :param test: If this plot is generated during test or validation
    """
    # Ensure the path `result` exists
    root_dir = get_root_dir()
    if not (os.path.exists(f'{root_dir}/result')):
        os.makedirs(f'{root_dir}/result')
    plot_other_metric_svm(hyper_param_name, 'precision', precision_lst, hyper_param_lst, test)
    plot_other_metric_svm(hyper_param_name, 'recall', recall_lst, hyper_param_lst, test)
    plot_other_metric_svm(hyper_param_name, 'f1', f1_lst, hyper_param_lst, test)
    plot_accuracy_svm(hyper_param_name, hyper_param_lst, accuracy_lst, test)


def plot_other_metric_cnn(model_name, epoch, lr, weight_decay, precision_lst, recall_lst, f1_lst):
    """Plots and saves a line chart for CNN experiment with three lines, where one line for precision, one for recall and
    the other one for f1

    :param model_name: Name of the model
    :param epoch: Total number of epochs
    :param lr: Learning rate
    :param weight_decay: Weight decay
    :param precision_lst: List of the precision value
    :param recall_lst: List of the recall value
    :param f1_lst: List of the f1 value
    """
    # Clear the previous content
    plt.clf()
    plt.cla()

    # Get the class names
    cls_lst = get_cls_name()
    x = range(len(cls_lst))

    # Plot the metrics
    plt.plot(x, precision_lst, marker='o', label='precision')
    plt.plot(x, recall_lst, marker='o', label='recall')
    plt.plot(x, f1_lst, marker='o', label='f1')

    # Show the legend
    plt.legend()
    # Show class names on the x-axis
    plt.xticks(x, cls_lst, rotation=45)

    # Save the plot
    root_dir = get_root_dir()
    plt.savefig(f'{root_dir}/result/{model_name}-ep{epoch}-lr{lr}-wd{weight_decay}-others.png')


def plot_metric_cnn(model_name, epoch, lr, weight_decay, lst, metric_name):
    """Plots and saves a line chart for CNN experiment with two lines, where one line for train set accuracy or loss and
    the other test set

    :param model_name: Name of the model
    :param epoch: Total number of epochs
    :param lr: Learning rate
    :param weight_decay: Weight decay
    :param lst: List of accuracy or loss, where the first element is from the train set and the second test set
    :param metric_name: Name of the metric
    """
    # Clear the previous content
    plt.clf()
    plt.cla()

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # Plot the metric
    epoch_lst = list(range(1, epoch + 1))
    train_lst, test_lst = lst
    plt.plot(epoch_lst, train_lst, marker='o', label='train')
    plt.plot(epoch_lst, test_lst, marker='o', label='test')

    # Show the legend
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel(metric_name)

    # Save the plot
    root_dir = get_root_dir()
    plt.savefig(f'{root_dir}/result/{model_name}-ep{epoch}-lr{lr}-wd{weight_decay}-{metric_name}.png')


def plot_result_cnn(model_name, epoch, lr, weight_decay, precision_lst, recall_lst, f1_lst, accuracy_lst, loss_lst):
    """Plots and saves the result for CNN experiment

    :param model_name: Name of the model
    :param epoch: Total number of epochs
    :param lr: Learning rate
    :param weight_decay: Weight decay
    :param precision_lst: List of the precision value
    :param recall_lst: List of the recall value
    :param f1_lst: List of the f1 value
    :param accuracy_lst: List of accuracy, where the first element is the train set accuracy and the second test set
    :param loss_lst: List of loss, where the first element is the train set accuracy and the second test set
    """
    # Ensure the path `result` exists
    root_dir = get_root_dir()
    if not (os.path.exists(f'{root_dir}/result')):
        os.makedirs(f'{root_dir}/result')
    plot_other_metric_cnn(model_name, epoch, lr, weight_decay, precision_lst, recall_lst, f1_lst)
    plot_metric_cnn(model_name, epoch, lr, weight_decay, accuracy_lst, 'accuracy')
    plot_metric_cnn(model_name, epoch, lr, weight_decay, loss_lst, 'loss')

if __name__ == '__main__':
    train_loader, test_loader = init_loader(full=True)
    train_data = next(iter(train_loader))
    test_data = next(iter(test_loader))
