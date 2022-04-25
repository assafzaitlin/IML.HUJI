from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        m, K, d = len(y), len(self.classes_), X.shape[1]
        self.pi_ = np.array([(y == i).sum() / m for i in self.classes_])
        self.mu_ = np.array([X[y == cls_].mean(axis=0) for cls_ in self.classes_])
        covs = []
        for i, cls_ in enumerate(self.classes_):
            samples = X[y == cls_]
            cov = np.zeros(d)
            mu = self.mu_[i]
            nk = len(samples)
            for sample in samples:
                centered = sample - mu
                cov += np.diag(np.outer(centered, centered))
            dividend = nk - K
            cov /= dividend
            covs.append(cov)
        self.vars_ = np.array(covs)

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
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        d = X.shape[1]
        const = -((d / 2) * np.log(2 * np.pi))
        likelihoods = []
        for sample in X:
            likelihood = []
            for i, pi in enumerate(self.pi_):
                var = self.vars_[i]
                calc = np.log(pi) - np.prod(var)
                centered = sample - self.mu_[i]
                inv = np.diagflat(1 / var)
                calc -= centered.T @ (inv @ centered)
                calc /= 2
                calc += const
                likelihood.append(calc)
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
