from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        m, K = len(y), len(self.classes_)
        self.pi_ = np.array([(y == i).sum() / m for i in self.classes_])
        self.mu_ = np.array([X[y == cls].mean(axis=0) for cls in self.classes_])
        cov = np.zeros(shape=(X.shape[1], X.shape[1]))
        for i, row in enumerate(X):
            mu = self.mu_[self.classes_ == y[i]]
            mat = row - mu
            res = np.matmul(mat.T, mat)
            cov += res
        self.cov_ = cov / (m - K)
        if det(self.cov_) == 0:
            raise ValueError("Matrix is not invertible")
        self._cov_inv = inv(self.cov_)

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
        likelihood = self.likelihood(X)
        argmax = np.argmax(likelihood, axis=1)
        return np.apply_along_axis(lambda x: self.classes_[x], 0, argmax)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling "
                             "`likelihood` function")
        likelihoods = []
        d = self.cov_.shape[0]
        const = (d / 2) * np.log(2 * np.pi) + 0.5 * det(self.cov_)
        const *= -1
        for row in X:
            likelihood = []
            sample_const = (row.T @ self._cov_inv) @ row
            for i, cls in enumerate(self.classes_):
                mu = self.mu_[i]
                a = self._cov_inv @ mu.T
                b = np.log(self.pi_[i]) - 0.5 * (mu @ self._cov_inv @ mu.T)
                likelihood.append((a @ row.T) + b + sample_const + const)
            likelihoods.append(likelihood)
        return np.array(likelihoods)


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
        from ...metrics import misclassification_error
        prediction = self.predict(X)
        return misclassification_error(y, prediction)
