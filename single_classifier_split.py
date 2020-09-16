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
        usage='Learns a single classifier over a specific dataset using split validation.')

    # Adds a dataset argument with pre-defined choices
    parser.add_argument('dataset', help='Dataset identifier', choices=[
                        'nsl-kdd', 'unespy'])

    # Adds an identifier argument to the desired classifier
    parser.add_argument('clf', help='Classifier identifier',
                        choices=['dt', 'linear_svc', 'lr', 'nb', 'opf', 'rf', 'svc'])

    # Adds an identifier argument to the desired size of the testing set
    parser.add_argument(
        '-test_size', help='Testing set size (between 0 and 1)', type=float, default=0.1)

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
    test_size = args.test_size
    n_runs = args.n_runs

    # Loads the data
    X, Y = l.load_dataset(dataset)

    # Running the classification task
    for i in range(n_runs):
        print(f'Running {i+1}/{n_runs} ...')

        # Performing the split
        X_train, X_test, Y_train, Y_test = l.use_split(
            X, Y, test_size=test_size, random_state=i)

        # Performing the full-learning process (training, testing and metrics)
        output = w.classify_split(clf, X_train, Y_train, X_test, Y_test)

        # Opening an output file
        with open(f'output/{dataset}_{args.clf}_split_{i}.pkl', 'wb') as dest_file:
            # Dumps the output object containing the metrics
            pickle.dump(output, dest_file)
