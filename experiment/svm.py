import math
import torch
import numpy as np

from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from utils import init_loader, get_data_once, plot_result


def apply_pca(amount, train_x, test_x):
    """Trains a PCA on X of train data and applies it on X of train and test data

    If the amount is 1, simply flatten the data without PCA processing
    :param amount: The least amount of variance that needs to be explained
    :param train_x: X of train data of shape B*C*H*W
    :param test_x: X of test data of shape B*C*H*W
    :return: The number of dimension used in PCA and X of train and test data after PCA
    """
    # If amount is equal to 1, flatten data to shape B*(C*H*W)
    if math.isclose(amount, 1, rel_tol=1e-4):
        new_train_x, new_test_x = torch.flatten(train_x, start_dim=1), torch.flatten(test_x, start_dim=1)
        dim = 3072
    # Process train_x and val_x with PCA if amount is not equal to 1
    else:
        # Flatten the data of shape B*C*H*W to B*(C*H*W)
        f_train_x = torch.flatten(train_x, start_dim=1)
        f_test_x = torch.flatten(test_x, start_dim=1)

        # Train a PCA with train data
        pca = PCA(n_components=amount, random_state=10)
        pca.fit(f_train_x)
        dim = pca.n_components_

        # Transform the data
        new_train_x = pca.transform(f_train_x)
        new_test_x = pca.transform(f_test_x)

    return dim, new_train_x, new_test_x


def cross_val_linear_svm(amount, fold, train_data):
    """Cross validation for a linear SVM trained on data whose dimension is reduced to a specific value with PCA

    :param amount: The least amount of variance that needs to be explained
    :param fold: Number of folds in cross validation
    :param train_data: Train data, a list of two tensors in which the first's shape is B*C*H*W and the second is B
    :return: Average number of dimensions used in PCA, precision, recall, f1 values (for each class) and accuracy of the SVM
    """
    # Lists that store the values of metrics in each round
    precision_lst = []
    recall_lst = []
    f1_lst = []
    accuracy_lst = []

    # List that stores the number of dimensions used in each round
    dim_lst = []

    # Init a linear SVM
    clf = svm.SVC(kernel='linear')

    # Cross validation
    for idx in range(fold):
        # Get the train data and val data for this round
        train_x, train_y, val_x, val_y = get_cross_val_data(fold, idx, train_data)

        # Process the data with PCA
        dim, train_x, val_x = apply_pca(amount, train_x, val_x)

        # Record the number of dimensions used in PCA in this round
        dim_lst.append(dim)

        # Train and evaluate the SVM
        precision, recall, f1, accuracy = train_eval_svm(clf, train_x, train_y, val_x, val_y)

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

    # Calculate the average number of dimensions used in PCA in all rounds
    avg_dim = round(np.mean(dim_lst, axis=0))

    print(f'Amount: {amount}, Dimension: {avg_dim}\nprecision: {avg_precision}\nrecall: {avg_recall}\nf1: {avg_f1}\naccuracy: {avg_accuracy}\n')

    return avg_dim, avg_precision, avg_recall, avg_f1, avg_accuracy


def cross_val_rbf_svm(c, fold, train_data):
    """Cross validation for a non-linear SVM with RGF kernal

    :param c: Regularization parameter for the SVM
    :param fold: Number of folds in cross validation
    :param train_data: Train data, a list of two tensors in which the first's shape is B*C*H*W and the second is B
    :return: Average precision, recall, f1 values (for each class) and accuracy of the SVM
    """
    # Lists that store the values of metrics in each round
    precision_lst = []
    recall_lst = []
    f1_lst = []
    accuracy_lst = []

    # Init a non-linear SVM with RBF kernal
    clf = svm.SVC(C=c)

    # Cross validation
    for idx in range(fold):
        # Get the train data and val data for this round
        train_x, train_y, val_x, val_y = get_cross_val_data(fold, idx, train_data)

        # Flatten the data of shape B*C*H*W to B*(C*H*W)
        train_x, val_x = torch.flatten(train_x, start_dim=1), torch.flatten(val_x, start_dim=1)

        # Train and evaluate the SVM
        precision, recall, f1, accuracy = train_eval_svm(clf, train_x, train_y, val_x, val_y)

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

    print(f'C: {c}\nprecision: {avg_precision}\nrecall: {avg_recall}\nf1: {avg_f1}\naccuracy: {avg_accuracy}\n')

    return avg_precision, avg_recall, avg_f1, avg_accuracy


def get_cross_val_data(fold, idx, train_data):
    """Separates the original train data into train data and val data for one round in cross validation

    :param fold: Number of folds for cross validation
    :param idx: Index of the current round
    :param train_data: Original train data
    :return: X, Y of train data and val data
    """
    # Calculate the number of data per fold
    num_total = train_data[0].shape[0]
    num_per_fold = math.ceil(num_total / fold)

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

    return train_x, train_y, val_x, val_y


def train_eval_svm(clf, train_x, train_y, test_x, test_y):
    """Trains and evaluates a SVM

    :param clf: A SVM
    :param train_x: X of train data
    :param train_y: Y of train data
    :param test_x: X of test data
    :param test_y: Y of test data
    :return: Precision, recall, f1 values (for each class) and accuracy of the SVM
    """
    # Train the SVM
    clf.fit(train_x, train_y)

    # Eval the SVM
    pred = clf.predict(test_x)
    precision = precision_score(test_y, pred, average=None)
    recall = recall_score(test_y, pred, average=None)
    f1 = f1_score(test_y, pred, average=None)
    accuracy = accuracy_score(test_y, pred)

    return precision, recall, f1, accuracy


def val_linear_svm(amount_lst, fold, train_data):
    """Evaluates performances of a series of linear SVMs trained on dimension-variant features

    The evaluation will be done on the val set.
    :param amount_lst: List of different amounts of variance that needs to be explained
    :param fold: Number of folds for cross validation
    :param train_data: Train data
    """
    # Lists that store the values of metrics for each selected amount
    precision_lst = []
    recall_lst = []
    f1_lst = []
    accuracy_lst = []

    # List that stores the number of dimension used in PCA for each selected amount
    dim_lst = []

    # Iterate through the selected dimensions and do cross validation
    for amount in amount_lst:
        dim, precision, recall, f1, accuracy = cross_val_linear_svm(amount, fold, train_data)

        # Record the values of metrics for this selected amount
        precision_lst.append(precision)
        recall_lst.append(recall)
        f1_lst.append(f1)
        accuracy_lst.append(accuracy)

        # Record the number of dimensions used in PCA for this selected amount
        dim_lst.append(dim)

    # Plot and save the result
    plot_result('dimension', dim_lst, precision_lst, recall_lst, f1_lst, accuracy_lst, False)


def test_linear_svm(amount_lst, train_data, test_data):
    """Evaluates performances of a series of linear SVMs trained on dimension-variant features

    The evaluation will be done on the test set.
    :param amount_lst: List of different amounts of variance that needs to be explained
    :param train_data: Train data
    :param train_data: Test data
    """
    # Lists that store the values of metrics for each selected amount
    precision_lst = []
    recall_lst = []
    f1_lst = []
    accuracy_lst = []

    # List that stores the number of dimension used in PCA for each selected amount
    dim_lst = []

    # Init a SVM
    clf = svm.SVC(kernel='linear')

    for amount in amount_lst:
        train_x = train_data[0]
        train_y = train_data[1]
        test_x = test_data[0]
        test_y = test_data[1]

        # Process the data with PCA
        dim, train_x, test_x = apply_pca(amount, train_x, test_x)

        # Train and evaluate the SVM
        precision, recall, f1, accuracy = train_eval_svm(clf, train_x, train_y, test_x, test_y)

        # Record the values of metrics for this selected amount
        precision_lst.append(precision)
        recall_lst.append(recall)
        f1_lst.append(f1)
        accuracy_lst.append(accuracy)

        # Record the number of dimensions used in PCA for this selected amount
        dim_lst.append(dim)

        print(f'Amount: {amount}, Dimension: {dim}\nprecision: {precision}\nrecall: {recall}\nf1: {f1}\naccuracy: {accuracy}\n')

    # Plot and save the result
    plot_result('dimension', dim_lst, precision_lst, recall_lst, f1_lst, accuracy_lst, True)


def val_rbf_svm(c_lst, fold, train_data):
    """Evaluates performances of a series of non-linear SVMs with RGF kernal

    The evaluation will be done on the val set.
    :param c_lst: List of different values of the regularization parameter C
    :param fold: Number of folds for cross validation
    :param train_data: Train data
    """
    # Lists that store the values of metrics for each selected C
    precision_lst = []
    recall_lst = []
    f1_lst = []
    accuracy_lst = []

    # Iterate through the selected dimensions and do cross validation
    for c in c_lst:
        precision, recall, f1, accuracy = cross_val_rbf_svm(c, fold, train_data)

        # Record the values of metrics for this selected amount
        precision_lst.append(precision)
        recall_lst.append(recall)
        f1_lst.append(f1)
        accuracy_lst.append(accuracy)

    # Plot and save the result
    plot_result('c', c_lst, precision_lst, recall_lst, f1_lst, accuracy_lst, False)


if __name__ == '__main__':
    train_loader, test_loader = init_loader(full=True)
    train_data = get_data_once(train_loader, 5000)
    test_data = get_data_once(test_loader, -1)

    start_amount = 1
    end_amount = 1.001
    step = 0.01

    amount_lst = np.arange(start_amount, end_amount, step)
    fold = 5
    # val_linear_svm(amount_lst, 5, train_data)
    # test_linear_svm(amount_lst, train_data, test_data)

    # cross_val_rbf_svm(10, 5, train_data)
    c_lst = [5, 10]
    val_rbf_svm(c_lst, 5, train_data)