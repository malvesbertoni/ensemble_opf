import argparse
import pickle

import utils.classifiers as c
import utils.load as l
import utils.wrapper as w
from models.stacking import Stacking


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(
        usage='Learns a stacking classifier over a specific dataset using k-folds cross validation.')

    # Adds a dataset argument with pre-defined choices
    parser.add_argument('dataset', help='Dataset identifier', choices=[
                        'nsl-kdd', 'unespy'])

    # Adds an identifier argument to the desired meta-learner
    parser.add_argument('meta', help='Meta-learner identifier',
                        choices=['dt', 'linear_svc', 'lr', 'nb', 'opf', 'opf_meta', 'rf', 'svc'])

    # Adds an identifier argument to the desired classifiers
    parser.add_argument('-c','--clfs', nargs='+', help='Classifier identifiers', required=True)

    # Adds an identifier argument to the desired size of the testing set
    parser.add_argument(
        '-test_size', help='Testing set size (between 0 and 1)', type=float, default=0.1)

    # Adds an identifier argument to the desired number of folds
    parser.add_argument(
        '-n_folds', help='Number of folds', type=int, default=10)

    # Adds an identifier argument to the desired number of runnings
    parser.add_argument(
        '-n_runs', help='Number of runnings', type=int, default=1)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering variables from arguments
    dataset = args.dataset
    meta = args.meta
    clfs = args.clfs
    test_size = args.test_size
    n_folds = args.n_folds
    n_runs = args.n_runs

    # Loads the data
    X, Y = l.load_dataset(dataset)

    # Creating the stacking-based ensemble
    stack = Stacking(clfs, meta)

    # Running the classification task
    for i in range(n_runs):
        print(f'Running {i+1}/{n_runs} ...')

        # Performing the split
        X_train, X_test, Y_train, Y_test = l.use_split(
            X, Y, test_size=test_size, random_state=i)

        # Gathering the folds
        folds = l.use_kfolds(X_train, Y_train, n_folds=n_folds,
                             random_state=i, shuffle=True)

        # Performing the full-learning process (training, testing and metrics)
        output = w.stacking_folds(stack, X_train, Y_train, folds, X_test, Y_test)

        # Creating an empty string to hold the base classifiers' names
        clf_str = ''

        # For every base classifier
        for clf in clfs:
            # Appends its name to the string
            clf_str += f'{clf}_'

        # Opening an output file
        with open(f'output/{dataset}_{clf_str}{meta}_stacking_{i}.pkl', 'wb') as dest_file:
            # Dumps the output object containing the metrics
            pickle.dump(output, dest_file)
