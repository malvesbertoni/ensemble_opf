import numpy as np

import utils.classifiers as c


class Stacking():
    """A stacking-based ensemble composed of two layers, i.e., a layer of base classifiers
        and a meta-learner layer.

    """

    def __init__(self, classifiers, meta):
        """Initialization method.

        Args:
            classifiers (list): List of classifiers' identifiers.
            meta (str): Meta-learner identifier.

        """

        # Creates an empty list of base classifiers
        self.clfs = []

        # Gathers the meta-learner class
        self.meta = c.get_clf(meta).obj

        # For every possible classifier
        for clf in classifiers:
            # Gathers the base classifier class
            self.clfs.append(c.get_clf(clf).obj)

    def fit_clfs(self, X_train, Y_train, X_val, Y_val):
        """Fits the base classifiers and returns their predictions.

        Args:
            X_train (np.array): Array of training samples.
            Y_train (np.array): Array of training labels.
            X_val (np.array): Array of validation samples.
            Y_val (np.array): Array of validation labels.

        Returns:
            The predictions out of the base classifiers along with the corresponding labels.
        """

        # Creates an empty array to hold the predictions
        preds = np.zeros((len(X_val), len(self.clfs)))

        # For every base classifier
        for i, clf in enumerate(self.clfs):
            # Trains the base classifier
            clf.fit(X_train, Y_train)

            # Transforms the validation set into predictions
            preds[:, i] = clf.predict(X_val)

        return preds, Y_val

    def fit_meta(self, X_train, Y_train):
        """Fits the meta-learner.

        Args:
            X_train (np.array): Array of training samples.
            Y_train (np.array): Array of training labels.

        """

        # Fits the meta-learner
        self.meta.fit(X_train, Y_train)

    def predict_clfs(self, X_test):
        """Predicts a sample into outputs using the base classifiers.

        Args:
            X_test (np.array): Array of testing samples.

        Returns:
            The predictions out of the base classifiers.

        """

        # Creates an empty array to hold the predictions
        preds = np.zeros((len(X_test), len(self.clfs)))

        # For every base classifier
        for i, clf in enumerate(self.clfs):
            # Transforms the testing set into predictions
            preds[:, i] = clf.predict(X_test)

        return preds

    def predict_meta(self, X_test):
        """Predicts a sample with the meta-learner.

        Args:
            X_test (np.array): Array of testing samples.

        Returns:
            The predictions out of the meta-learner.

        """

        # Predicts using the meta-learner
        preds = self.meta.predict(X_test)

        return preds
