from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    noiseless_x = np.linspace(-1.2, 2, n_samples)
    df = pd.DataFrame(noiseless_x, columns=['X'])
    df['y'] = df['X'].apply(lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2))
    copy = df.copy()
    noise = np.random.normal(0, noise, n_samples)
    copy['y'] = copy['y'].to_numpy() + noise
    X, y = copy[['X']], copy['y']
    train_X, train_y, test_X, test_y = split_train_test(X, y,
                                                        train_proportion=2/3)
    train_samples = train_X.copy()
    train_samples['y'] = train_y
    test_samples = test_X.copy()
    test_samples['y'] = test_y
    df['type'] = 'original samples'
    train_samples['type'] = 'train samples'
    test_samples['type'] = 'test samples'
    all_samples = pd.concat([df, test_samples, train_samples])
    px.scatter(all_samples, x='X', y='y', color='type',
               title="Origin, test & train samples").show()
    # concat dataframes with string indicating if train/ test/ noise-less

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_errors = []
    validation_errors = []
    for deg in range(11):
        train_err, validation_err = cross_validate(PolynomialFitting(deg),
                                                   train_X.to_numpy(),
                                                   train_y.to_numpy(),
                                                   mean_square_error)
        train_errors.append((deg, train_err))
        validation_errors.append((deg, validation_err))
    df1 = pd.DataFrame(train_errors, columns=['degree of polynomial', 'error'])
    df2 = pd.DataFrame(validation_errors, columns=['degree of polynomial', 'error'])
    df1['error type'] = 'train'
    df2['error type'] = 'validation'
    df = pd.concat([df1, df2])
    px.line(df, x='degree of polynomial', y='error', color='error type',
            title='Error based on type of samples and degree of fitted polynomial').show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    validation_errors = np.array(validation_errors)
    best_error_deg = np.argmin(validation_errors, axis=0)[1]
    best_degree = int(validation_errors[best_error_deg][0])
    best_error = round(validation_errors[best_error_deg][1], 2)
    estimator = PolynomialFitting(best_degree)
    estimator.fit(train_X.to_numpy(), train_y.to_numpy())
    test_err = estimator.loss(test_X.to_numpy(), test_y.to_numpy())
    test_err = round(test_err, 2)
    print(f"best polynomial degree: {best_degree}. test error: {test_err}. best error: {best_error}")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    raise NotImplementedError()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    raise NotImplementedError()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
