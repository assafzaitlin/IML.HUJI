from __future__ import annotations
from typing import NoReturn
from IMLearn.base import BaseEstimator
import numpy as np
from sklearn.linear_model import LinearRegression


class AgodaCancellationEstimator(BaseEstimator):
    """
    An estimator for solving the Agoda Cancellation challenge
    """

    def __init__(self) -> AgodaCancellationEstimator:
        """
        Instantiate an estimator for solving the Agoda Cancellation challenge
        Parameters
        ----------
        Attributes
        ----------
        """
        super().__init__()
        self.model = LinearRegression()


    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an estimator for given samples
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for
        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        Notes
        -----
        """
        self.model.fit(X, y)

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
        """
        return np.array([1 if i >= 0.5 else 0 for i in self.model.predict(X)])

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under loss function
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples
        y : ndarray of shape (n_samples, )
            True labels of test samples
        Returns
        -------
        loss : float
            Performance under loss function
        """
        res = self.predict(X)
        num_of_errors = 0
        for index, i in enumerate(res):
            if y[index] != i:
                num_of_errors += 1
        print(f"num of errors: {num_of_errors}, sample size: {len(res)}")
        return num_of_errors / len(res)