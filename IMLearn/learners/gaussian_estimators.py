from __future__ import annotations
from numpy.linalg import inv, det, slogdet
import numpy as np


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """
    def __init__(self, biased_var: bool = False) -> None:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = X.mean()
        self.var_ = sum((X - self.mu_) ** 2) / (X.size - 1)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        exp_power = -1 * ((X - self.mu_) ** 2 / (2 * self.var_))
        return (1 / (2 * np.pi * self.var_) ** 0.5) * np.e ** exp_power

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        likelihood_sum = 0
        log_2pi = np.log(2 * np.pi)
        log_sigma = np.log(sigma)
        for sample in X:
            normalized_sample = sample - mu
            exp_power = normalized_sample ** 2 / sigma
            likelihood = -0.5 * (exp_power + log_2pi + log_sigma)
            likelihood_sum += likelihood
        return likelihood_sum


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """
    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = np.mean(X, axis=0)
        self.cov_ = np.cov(X, rowvar=False)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        normalized = X - self.mu_
        cov_inverse = np.linalg.inv(self.cov_)
        exp_power = -0.5 * np.matmul(np.matmul(normalized, cov_inverse),
                                     normalized.transpose())
        d = len(self.mu_)
        det_cov = np.linalg.det(self.cov_)
        results = (1 / ((2 * np.pi) ** d * det_cov)) * np.e ** exp_power
        return results.sum(axis=0)

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """
        cov_inverse = np.linalg.inv(cov)
        cov_log_det = slogdet(cov)
        cov_log_det = cov_log_det[0] * cov_log_det[1]
        d = len(mu)
        log_2pi = np.log(2 * np.pi)
        dlog_2pi = d * log_2pi
        normalized_samples = X - mu
        mul_result = np.sum(normalized_samples @ cov_inverse * normalized_samples)
        samples_num = len(X)
        return -0.5 * (samples_num * dlog_2pi - samples_num * cov_log_det + mul_result)
        # for sample in X:
        #     normalized_sample = sample - mu
        #     log_exp_power = np.matmul(np.matmul(normalized_sample.transpose(),
        #                                         cov_inverse),
        #                               normalized_sample)
        #     if isinstance(log_exp_power, np.ndarray):
        #         log_exp_power = log_exp_power.sum()
        #     log_likelihood = -0.5 * (cov_log_det + dlog_2pi + log_exp_power)
        #     log_likelihood_sum += log_likelihood
        # return log_likelihood_sum
