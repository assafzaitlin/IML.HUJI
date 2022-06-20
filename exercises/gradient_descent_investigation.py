import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from IMLearn.model_selection.cross_validate import cross_validate
from IMLearn.metrics.loss_functions import misclassification_error

import plotly.graph_objects as go
import plotly.express as px


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    weights = []

    def callback(**kwargs):
        weights.append(kwargs['weights'])
        values.append(kwargs['val'])

    return callback, values, weights


def plot_delta(values, title, names=None):
    if names is None:
        scatters = [go.Scatter(y=val, x=list(range(len(val))), mode='lines')
                    for val in values]
    else:
        scatters = [go.Scatter(y=values[i], x=list(range(len(values[i]))),
                               name=names[i], mode='lines')
                    for i in range(len(values))]
    fig = go.Figure(scatters, layout=go.Layout(title=title))
    fig.update_xaxes(title="Num of iterations").update_yaxes(title='Loss')
    return fig

def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    losses = np.zeros((len(etas), 2))
    l1_deltas = []
    l1_names = []
    l2_deltas = []
    l2_names = []
    for i, eta in enumerate(etas):
        lr = FixedLR(eta)
        callback, values, weights = get_gd_state_recorder_callback()
        l1 = L1(init)
        gd = GradientDescent(learning_rate=lr, callback=callback)
        gd.fit(l1, None, None)
        plot = plot_descent_path(module=L1, descent_path=np.array(weights),
                                 title=f"L1 descent path with eta: {eta}")
        plot.show()
        l1_deltas.append(values.copy())
        l1_names.append(f"eta={eta}")
        min_loss = np.min(values)
        losses[i][0] = min_loss
        values.clear()
        weights.clear()
        l2 = L2(init)
        gd.fit(l2, None, None)
        plot = plot_descent_path(module=L2, descent_path=np.array(weights),
                                 title=f"L2 descent path with eta: {eta}")
        plot.show()
        l2_deltas.append(values)
        l2_names.append(f"eta={eta}")
        min_loss = np.min(values)
        losses[i][1] = min_loss
    min_ind_l1, min_ind_l2 = np.argmin(losses, axis=0)
    print(f"best eta for L1 is {etas[min_ind_l1]} with loss {losses[min_ind_l1][0]}")
    print(f"best eta for L2 is {etas[min_ind_l2]} with loss {losses[min_ind_l2][1]}")
    l1_graph = plot_delta(l1_deltas, "Loss as function of number of iterations"
                                     " for L1", l1_names)
    l1_graph.show()
    l2_graph = plot_delta(l2_deltas, "Loss as function of number of iterations"
                                     " for L2", l2_names)
    l2_graph.show()

def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    vals = []
    min_loss = np.zeros(len(gammas))
    decision_surface = decision_surface2 = None
    for i, gamma in enumerate(gammas):
        lr = ExponentialLR(eta, gamma)
        callback, values, weights = get_gd_state_recorder_callback()
        l1 = L1(init)
        gd = GradientDescent(learning_rate=lr, callback=callback)
        gd.fit(l1, None, None)
        vals.append(values)
        min_loss[i] = np.min(values)
        if gamma == 0.95:
            decision_surface = plot_descent_path(module=L1,
                                                 descent_path=np.array(weights.copy()),
                                                 title="descent path with "
                                                       "gamma=0.95 for L1")
            l2 = L2(init)
            values.clear()
            weights.clear()
            gd = GradientDescent(learning_rate=lr, callback=callback)
            gd.fit(l2, None, None)
            decision_surface2 = plot_descent_path(module=L2,
                                                  descent_path=np.array(weights),
                                                  title="descent path with "
                                                        "gamma=0.95 for L2")
    names = [f"gamma: {gamma}" for gamma in gammas]
    fig = plot_delta(vals, "Loss as num of iterations for all gammas", names)
    fig.show()

    # Plot algorithm's convergence for the different values of gamma
    min_ind = int(np.argmin(min_loss))
    best_gamma = gammas[min_ind]
    best_loss = min_loss[min_ind]
    print(f"Best loss is with gamma {best_gamma}. loss is {best_loss}")

    # Plot descent path for gamma=0.95
    decision_surface.show()
    decision_surface2.show()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    alphas = np.linspace(0, 1, 101)
    model = LogisticRegression()
    model.fit(X=X_train.to_numpy(), y=y_train.to_numpy())
    pred = model.predict_proba(X_train.to_numpy())
    tpr = []
    fpr = []
    diff = []
    for alpha in alphas:
        y_pred = np.where(pred >= alpha, 1, 0)
        fp = np.sum((y_pred == 1) & (y_train == 0))
        tp = np.sum((y_pred == 1) & (y_train == 1))
        fn = np.sum((y_pred == 0) & (y_train == 1))
        tn = np.sum((y_pred == 0) & (y_train == 0))
        fpr_calc = fp / (fp + tn)
        tpr_calc = tp / (tp + fn)
        fpr.append(fpr_calc)
        tpr.append(tpr_calc)
        diff.append(tpr_calc - fpr_calc)
    scatter = go.Scatter(x=fpr, y=tpr, mode='lines')
    fig = go.Figure([scatter], layout=go.Layout(title="ROC for all alpha values"))
    fig.update_xaxes(title="False positive").update_yaxes(title="True positive")
    fig.show()
    best_alpha = np.argmax(diff) / 100
    model = LogisticRegression(alpha=best_alpha)
    model.fit(X_train.to_numpy(), y_train.to_numpy())
    loss = model.loss(X_test.to_numpy(), y_test.to_numpy())
    print(f"best alpha is: {best_alpha} with error {loss}")

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lambdas = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
    for i in range(1, 3):
        validation_scores = np.zeros(lambdas.shape[0])
        train_scores = np.zeros(lambdas.shape[0])
        for j, lam in enumerate(lambdas):
            model = LogisticRegression(penalty=f"l{i}", lam=lam)
            train_score, validation_score = cross_validate(model,
                                                           X_train.to_numpy(),
                                                           y_train.to_numpy(),
                                                           misclassification_error)
            validation_scores[j] = validation_score
            train_scores[j] = train_score
        best_lambda = lambdas[np.argmin(validation_scores)]
        model = LogisticRegression(penalty=f"l{i}", lam=best_lambda)
        model.fit(X_train.to_numpy(), y_train.to_numpy())
        loss = model.loss(X_test.to_numpy(), y_test.to_numpy())
        print(f"Best lambda for model with regularization L{i} is: {loss} with lambda {best_lambda}")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
