import time
import numpy as np

from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)


def classify_folds(clf, X, Y, folds):
    """Performs the full-learning procedure (training, testing and metrics).

    Args:
        clf (Classifier): A Classifier instance.
        X (np.array): Array of samples.
        Y (np.array): Array of labels.
        folds (list): A list of k-folds indexes.

    Returns:
        A dictionary holding the classification metrics.

    """

    # Defining lists for further appending
    c_matrix, accuracy, precision, recall, f1, _time = [], [], [], [], [], []

    # Iterating through every fold
    for j, (train_idx, test_idx) in enumerate(folds):
        print(f'Running fold {j+1} ...')

        # Applying fold's indexes to training and testing sets
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        # Starting timer
        start = time.time()

        # Fitting classifier with training data
        clf.fit(X_train, Y_train)

        # Ending timer
        end = time.time()

        # Predicting test data
        preds = clf.predict(X_test)

        # Calculating the desired metrics
        c_matrix.append(confusion_matrix(Y_test, preds))
        accuracy.append(accuracy_score(Y_test, preds))
        precision.append(precision_score(Y_test, preds))
        recall.append(recall_score(Y_test, preds))
        f1.append(f1_score(Y_test, preds))
        _time.append((end - start))

    return {
        'c_matrix': c_matrix,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'time': _time
    }


def classify_split(clf, X_train, Y_train, X_test, Y_test):
    """Performs the full-learning procedure (training, testing and metrics).

    Args:
        clf (Classifier): A Classifier instance.
        X_train (np.array): Array of training samples.
        Y_train (np.array): Array of training labels.
        X_test (np.array): Array of testing samples.
        Y_test (np.array): Array of testing labels.

    Returns:
        A dictionary holding the classification metrics.

    """

    # Starting timer
    start = time.time()

    # Fitting classifier with training data
    clf.fit(X_train, Y_train)

    # Ending timer
    end = time.time()

    # Predicting test data
    preds = clf.predict(X_test)

    # Calculating the desired metrics
    c_matrix = confusion_matrix(Y_test, preds)
    accuracy = accuracy_score(Y_test, preds)
    precision = precision_score(Y_test, preds)
    recall = recall_score(Y_test, preds)
    f1 = f1_score(Y_test, preds)
    _time = end - start

    return {
        'c_matrix': c_matrix,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'time': _time
    }


def stacking_folds(stack, X_train, Y_train, folds, X_test, Y_test):
    """Performs the full-learning procedure over a stacking-based ensemble.

    Args:
        stack (Stacking): A Stacking instance.
        X_train (np.array): Array of training samples.
        Y_train (np.array): Array of training labels.
        folds (list): A list of k-folds indexes.
        Y_train (np.array): Array of training labels.
        Y_test (np.array): Array of testing labels.

    Returns:
        A dictionary holding the classification metrics.

    """

    # Starting timer
    start = time.time()

    # Iterating through every fold
    for j, (train_idx, val_idx) in enumerate(folds):
        print(f'Running fold {j+1} ...')

        # Applying fold's indexes to training and validation sets
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        Y_tr, Y_val = Y_train[train_idx], Y_train[val_idx]

        # Fit the base classifiers
        X_folds, Y_folds = stack.fit_clfs(X_tr, Y_tr, X_val, Y_val)

        # If it is the first fold
        if j == 0:
            # Gathers the prediction out of the fitted classifiers
            X_train_stacked = X_folds

            # Gathers the labels out of the fitted classifiers
            Y_train_stacked = Y_folds

        # If it is not the first fold
        else:
            # Stacks the array to compose the meta-learner training set samples
            X_train_stacked = np.vstack((X_train_stacked, X_folds))

            # Stacks the array to compose the meta-learner training set labels
            Y_train_stacked = np.hstack((Y_train_stacked, Y_folds))

    # Predicts with the base classifiers the meta-learner testing set samples
    X_test_stacked = stack.predict_clfs(X_test)

    # Gathers the meta-learner testing set samples
    Y_test_stacked = Y_test
    
    # Little trick to satisfy OPF `canberra` distance function
    # Note that it will not affect other classifiers
    X_train_stacked -= 1
    X_test_stacked -= 1

    # Fits the meta-learner
    stack.fit_meta(X_train_stacked, Y_train_stacked)

    # Ending timer
    end = time.time()

    # Predicts with the meta-learner
    preds = stack.predict_meta(X_test_stacked)

    # Calculating the desired metrics
    c_matrix = confusion_matrix(Y_test_stacked, preds)
    accuracy = accuracy_score(Y_test_stacked, preds)
    precision = precision_score(Y_test_stacked, preds)
    recall = recall_score(Y_test_stacked, preds)
    f1 = f1_score(Y_test_stacked, preds)
    _time = end - start

    return {
        'c_matrix': c_matrix,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'time': _time
    }
