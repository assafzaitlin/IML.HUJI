from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        loss_per_feature = np.array([self._find_threshold(feature, y, 1) for
                                     feature in X.T])
        loss_per_feature_negative = np.array([self._find_threshold(feature, y,
                                                                   -1)
                                              for feature in X.T])
        minimal_ind = np.argmin(loss_per_feature, axis=0)[0]
        minimal_ind_negative = np.argmin(loss_per_feature_negative, axis=0)[1]
        minimal_loss = loss_per_feature[minimal_ind]
        minimal_loss_negative = loss_per_feature_negative[minimal_ind_negative]
        if minimal_loss[1] <= minimal_loss_negative[1]:
            self.j_ = minimal_ind
            self.sign_ = 1
            self.threshold_ = minimal_loss[0]
        else:
            self.j_ = minimal_ind_negative
            self.sign_ = -1
            self.threshold_ = minimal_loss_negative[0]

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        func = lambda x: self.sign_ if x[self.j_] >= self.threshold_ else -1 * self.sign_
        return np.apply_along_axis(func, 1, X)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassification error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        sorted_indices = values.argsort()
        sorted_values = values[sorted_indices]
        sorted_labels = labels[sorted_indices]
        assignment = np.ones(labels.shape[0])
        if sign < 0:
            assignment *= -1
        losses = np.zeros((labels.shape[0], 2))
        for i, threshold in enumerate(sorted_values):
            loss = np.sum(np.abs(sorted_labels[np.sign(sorted_labels) !=
                                               np.sign(assignment)]
                                 )) / assignment.shape[0]
            losses[i] = (threshold, loss)
            assignment[i] *= -1
        return losses[np.argmin(losses, axis=0)[1]]

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        predicted = self.predict(X)
        loss = np.sum(np.abs(y[np.sign(y) != np.sign(predicted)])) / y.shape[0]
        return loss
