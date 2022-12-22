import math
import torch
import numpy as np

from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from utils import init_loader, get_data_once, plot_multi_line, plot_single_line


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
    pca = PCA(n_components=dim, random_state=10)
    pca.fit(f_train_x)

    # Transform the data
    new_train_x = pca.transform(f_train_x)
    new_test_x = pca.transform(f_test_x)

    return new_train_x, new_test_x


def cross_val_pca(dim, fold, train_data):
    """Cross validation for a SVM trained on data whose dimension is reduced to a specific value with PCA

    :param dim: Dimension of the data after PCA
    :param fold: Number of folds in cross validation
    :param train_data: Train data, a list of two tensors in which the first's shape is B*C*H*W and the second is B
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
    num_total = train_data[0].shape[0]
    num_per_fold = math.ceil(num_total / fold)

    # Cross validation
    for idx in range(fold):
        # Get the train data and val data for this round
        # If the selected val set is not the last fold
        if idx != fold - 1:
            val_x = train_data[0][idx * num_per_fold:(idx + 1) * num_per_fold]
            val_y = train_data[1][idx * num_per_fold:(idx + 1) * num_per_fold]
            train_x = torch.cat((train_data[0][:idx * num_per_fold], train_data[0][(idx + 1) * num_per_fold:]), 0)
            train_y = torch.cat((train_data[1][:idx * num_per_fold], train_data[1][(idx + 1) * num_per_fold:]), 0)
        # If the selected val set is the last fold
        else:
            val_x = train_data[0][idx * num_per_fold:]
            val_y = train_data[1][idx * num_per_fold:]
            train_x = train_data[0][:idx * num_per_fold]
            train_y = train_data[1][:idx * num_per_fold]

        # Process train_x and val_x with PCA if dim is not equal to 3072
        if dim != 3072:
            train_x, val_x = apply_pca(dim, train_x, val_x)
        # If dim is equal to 3072, flatten data to shape B*(C*H*W)
        else:
            train_x, val_x = torch.flatten(train_x, start_dim=1), torch.flatten(val_x, start_dim=1)

        assert train_x.shape[1] == dim and val_x.shape[1] == dim, 'Something wrong when dealing with the dimension'

        # Train the SVM
        clf.fit(train_x, train_y)

        # Eval the SVM
        pred = clf.predict(val_x)
        precision = precision_score(val_y, pred, average=None)
        recall = recall_score(val_y, pred, average=None)
        f1 = f1_score(val_y, pred, average=None)
        accuracy = accuracy_score(val_y, pred)

        # Record the values of metrics in this round
        precision_lst.append(precision)
        recall_lst.append(recall)
        f1_lst.append(f1)
        accuracy_lst.append(accuracy)
        # print(f'Round {idx + 1}\nprecision: {precision}\nrecall: {recall}\nf1: {f1}\naccuracy: {accuracy}\n')

    # Calculate average values of metrics in all rounds
    avg_precision = np.mean(precision_lst, axis=0)
    avg_recall = np.mean(recall_lst, axis=0)
    avg_f1 = np.mean(f1_lst, axis=0)
    avg_accuracy = np.mean(accuracy_lst, axis=0)
    print(f'Average\nprecision: {avg_precision}\nrecall: {avg_recall}\nf1: {avg_f1}\naccuracy: {avg_accuracy}\n')

    return avg_precision, avg_recall, avg_f1, avg_accuracy


def val_linear_svm(dims, fold, train_data):
    """Evaluates performances of a series of linear SVMs trained on dimension-variant features

    The evaluation will be done on the val set.
    :param dims: List of different dimensions
    :param fold: Number of folds for cross validation
    :param train_data: Train data
    """
    # Lists that store the values of metrics for each selected dimension
    precision_lst = []
    recall_lst = []
    f1_lst = []
    accuracy_lst = []

    # Iterate through the selected dimensions and do cross validation
    for dim in dims:
        precision, recall, f1, accuracy = cross_val_pca(dim, fold, train_data)

        # Record the values of metrics for this selected dimension
        precision_lst.append(precision)
        recall_lst.append(recall)
        f1_lst.append(f1)
        accuracy_lst.append(accuracy)

    # Plot and save the result
    plot_multi_line('precision', precision_lst, dims, False)
    plot_multi_line('recall', recall_lst, dims, False)
    plot_multi_line('f1', f1_lst, dims, False)
    plot_single_line(accuracy_lst, dims, False)


if __name__ == '__main__':
    train_loader, test_loader = init_loader(full=True)
    train_data = get_data_once(train_loader, 5000)
    test_data = get_data_once(test_loader, -1)

    dims = [3072, 2000]
    fold = 5
    val_linear_svm(dims, 5, train_data)
