import argparse
import pickle

import utils.classifiers as c
import utils.load as l
import utils.wrapper as w


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(
        usage='Learns a single classifier over a specific dataset using k-folds cross validation.')

    # Adds a dataset argument with pre-defined choices
    parser.add_argument('dataset', help='Dataset identifier', choices=[
                        'nsl-kdd', 'unespy'])

    # Adds an identifier argument to the desired classifier
    parser.add_argument('clf', help='Classifier identifier',
                        choices=['dt', 'linear_svc', 'lr', 'nb', 'opf', 'rf', 'svc'])

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
    clf = c.get_clf(args.clf).obj
    n_folds = args.n_folds
    n_runs = args.n_runs

    # Loads the data
    X, Y = l.load_dataset(dataset)

    # Running the classification task
    for i in range(n_runs):
        print(f'Running {i+1}/{n_runs} ...')

        # Gathering the folds
        folds = l.use_kfolds(X, Y, n_folds=n_folds,
                             random_state=i, shuffle=True)

        # Performing the full-learning process (training, testing and metrics)
        output = w.classify_folds(clf, X, Y, folds)

        # Opening an output file
        with open(f'output/{dataset}_{args.clf}_kfolds_{i}.pkl', 'wb') as dest_file:
            # Dumps the output object containing the metrics
            pickle.dump(output, dest_file)
