import argparse
import pickle

import numpy as np


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(
        usage='Process the saved metrics file into real results.')

    # Adds a dataset argument with pre-defined choices
    parser.add_argument('dataset', help='Dataset identifier', choices=[
                        'nsl-kdd', 'unespy'])

    # Adds an identifier argument to the desired classifier
    parser.add_argument('clf', help='Classifier identifier',
                        choices=['dt', 'linear_svc', 'lr', 'nb', 'opf', 'rf', 'svc'])

    # Adds an identifier argument to the desired type of cross validation
    parser.add_argument('type', help='Type of cross validation', choices=[
                        'split', 'kfolds'])

    # Adds an identifier argument to the desired running identifier
    parser.add_argument('-run', help='Running identifier', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    dataset = args.dataset
    clf = args.clf
    type = args.type
    run = args.run

    # Defining an input file
    input_file = f'output/{dataset}_{clf}_{type}_{run}.pkl'

    # Trying to open the file
    with open(input_file, "rb") as origin_file:
        # Loading model from file
        metrics = pickle.load(origin_file)

    # Checks if cross validation is k-folds
    if type == 'kfolds':
        # If yes, gathers the mean of confusion matrix
        mean_c_matrix = np.mean(metrics['c_matrix'], axis=0)

        # If yes, gathers the standard deviation of confusion matrix
        std_c_matrix = np.std(metrics['c_matrix'], axis=0)

    # If it is a split validation
    elif type == 'split':
        # Confusion matrix can be gathered right away
        mean_c_matrix = metrics['c_matrix']

        # Confusion matrix can be gathered right away
        std_c_matrix = np.zeros((2, 2))

    # Gathering metrics' mean values
    mean_accuracy = np.mean(metrics['accuracy'])
    mean_precision = np.mean(metrics['precision'])
    mean_recall = np.mean(metrics['recall'])
    mean_f1 = np.mean(metrics['f1'])
    mean_time = np.mean(metrics['time'])
    total_time = np.sum(metrics['time'])

    # Gathering metrics' standard deviation values
    std_accuracy = np.std(metrics['accuracy'])
    std_precision = np.std(metrics['precision'])
    std_recall = np.std(metrics['recall'])
    std_f1 = np.std(metrics['f1'])
    std_time = np.std(metrics['time'])

    print('\nSaving outputs ...')

    with open(f'output/{dataset}_{clf}_{type}_{run}_mean.txt', 'w') as f:
        f.write(f'{mean_c_matrix} {mean_accuracy} {mean_precision} {mean_recall} {mean_f1} {mean_time} {total_time}')

    with open(f'output/{dataset}_{clf}_{type}_{run}_std.txt', 'w') as f:
        f.write(f'{std_c_matrix} {std_accuracy} {std_precision} {std_recall} {std_f1} {std_time}')

    print('Outputs saved.')
