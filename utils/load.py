import opfython.stream.loader as l
import opfython.stream.parser as p
import opfython.stream.splitter as s
from sklearn.model_selection import KFold


def load_dataset(dataset='NSL-KDD'):
    """Loads a dataset.

    Args:
        dataset (str): Dataset's identifier.

    Returns:
        X and Y (samples and labels).

    """

    # If the dataset is `nslkdd`
    if dataset == 'nsl-kdd':
        # Loading a .txt file to a numpy array
        txt = l.load_txt('data/nsl-kdd.txt')

    # If the dataset is `unespy`
    elif dataset == 'unespy':
        # Loading a .txt file to a numpy array
        txt = l.load_txt('data/unespy.txt')

    # Parsing a pre-loaded numpy array
    X, Y = p.parse_loader(txt)

    return X, Y


def use_split(X, Y, test_size=0.1, random_state=0):
    """Splits the data into training and testing sets.

    Args:
        X (np.array): Input samples array.
        Y (np.array): Input labels array.
        test_size (float): Size of the testing set.
        random_state (int): Random integer to provide a random state to splitter.

    """

    # Splitting data into training and testing sets
    X_train, X_test, Y_train, Y_test = s.split(
        X, Y, percentage=1-test_size, random_state=random_state)

    return X_train, X_test, Y_train, Y_test


def use_kfolds(X, Y, n_folds=10, random_state=0, shuffle=True):
    """Splits X, Y (samples, labels) into k-folds.

    Args:
        X (np.array): Input samples array.
        Y (np.array): Input labels array.
        n_folds (int): Number of folds.
        random_state (int): Random integer to provide a random state to splitter.
        shuffle (bool): Whether the data should be shuffled or not.

    Returns:
        The k-fold indexes.

    """

    # Creating K-Folds object
    kf = KFold(n_folds, shuffle, random_state)

    # Gathering the folds' indexes
    idx_folds = kf.split(X, Y)

    return idx_folds
