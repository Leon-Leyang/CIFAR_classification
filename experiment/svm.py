import math
import torch
import numpy as np

from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


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


def cross_val_pca(dim, fold, full_train_data):
    """Cross validation for a SVM trained on data whose dimension is reduced to a specific value with PCA

    :param dim: Dimension of the data after PCA
    :param fold: Number of folds in cross validation
    :param full_train_data: Train data, a list of two tensors in which the first's shape is B*C*H*W and the second is B
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


if __name__ == '__main__':
    pass