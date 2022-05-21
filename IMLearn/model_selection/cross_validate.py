from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    folds = np.array_split(X, cv, axis=0)
    results_folds = np.array_split(y, cv, axis=0)
    train_score = 0
    validation_score = 0
    for i in range(cv):
        test_X, test_y = folds[i], results_folds[i]
        train_X = [folds[j] for j in range(cv) if j != i]
        train_y = [results_folds[j] for j in range(cv) if j != i]
        train_X = np.concatenate(train_X, axis=0)
        train_y = np.concatenate(train_y, axis=0)
        estimator.fit(train_X, train_y)
        train_score += scoring(train_y, estimator.predict(train_X))
        validation_score += scoring(test_y, estimator.predict(test_X))
    return train_score / cv, validation_score / cv
