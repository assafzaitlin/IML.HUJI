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
    print(f"best polynomial degree: {best_degree}. test error: {test_err}")


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
    X, y = datasets.load_diabetes(return_X_y=True)
    train_X, train_y = X[:n_samples], y[:n_samples]
    test_X, test_y = X[n_samples:], y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambdas = np.linspace(0, 2.5, n_evaluations)
    lasso_train_errors = np.zeros(n_evaluations)
    lasso_validation_errors = np.zeros(n_evaluations)
    ridge_train_errors = np.zeros(n_evaluations)
    ridge_validation_errors = np.zeros(n_evaluations)
    for i, l in enumerate(lambdas):
        estimator1 = RidgeRegression(l)
        ridge_train_err, ridge_validation_err = cross_validate(estimator1, train_X,
                                                         train_y,
                                                         mean_square_error)
        estimator2 = Lasso(l)
        lasso_train_err, lasso_validation_err = cross_validate(estimator2, train_X,
                                                         train_y,
                                                         mean_square_error)
        ridge_train_errors[i] = ridge_train_err
        ridge_validation_errors[i] = ridge_validation_err
        lasso_validation_errors[i] = lasso_validation_err
        lasso_train_errors[i] = lasso_train_err

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=ridge_train_errors, x=lambdas,
                             mode='lines+markers', name='ridge train error'))
    fig.add_trace(go.Scatter(y=ridge_validation_errors, x=lambdas,
                             mode='lines+markers', name='ridge validation error'))
    fig.add_trace(go.Scatter(y=lasso_train_errors, x=lambdas,
                             mode='lines+markers', name='lasso train error'))
    fig.add_trace(go.Scatter(y=lasso_validation_errors, x=lambdas,
                             mode='lines+markers', name='lasso validation error'))
    fig.update_layout(title="Error of Ridge & Lasso regressors")
    fig.update_xaxes(title='lambda value')
    fig.update_yaxes(title='MSE')
    fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    ridge_best_lambda = lambdas[np.argmin(ridge_validation_errors, axis=0)]
    lasso_best_lambda = lambdas[np.argmin(lasso_validation_errors, axis=0)]
    ridge = RidgeRegression(ridge_best_lambda).fit(train_X, train_y)
    lasso = Lasso(lasso_best_lambda).fit(train_X, train_y)
    ls = LinearRegression().fit(train_X, train_y)
    ridge_loss = round(ridge.loss(test_X, test_y), 3)
    lasso_loss = round(mean_square_error(lasso.predict(test_X), test_y), 3)
    ls_loss = round(ls.loss(test_X, test_y), 3)
    ridge_best_lambda = round(ridge_best_lambda, 3)
    lasso_best_lambda = round(lasso_best_lambda, 3)
    print(f"LS best error: {ls_loss}, "
          f"Ridge error: {ridge_loss} with optimal lambda: {ridge_best_lambda}, "
          f"Lasso error: {lasso_loss} with optimal lambda: {lasso_best_lambda}")


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
