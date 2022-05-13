import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metrics.loss_functions import accuracy
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def create_decision_surface_scatter(model: AdaBoost, size: int, X: np.array,
                                    y: np.array, symbols: np.array,
                                    limits: np.array, weights=None):
    """
    @param model: The model to predict by
    @param size: number of base learners to use
    @param X: The samples
    @param y: The results
    @param symbols: symbols to give to results
    @param limits: The limits of the surface
    @param weights: Weights for points in scatter plot
    Returns decision surface and scatter graph based on the parameters
    """
    predict_func = lambda x: model.partial_predict(x, size)
    surface = decision_surface(predict_func, limits[0], limits[1],
                               showscale=False)
    if weights is None:
        graph = go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                           marker=dict(color=y.astype(int),
                                       symbol=symbols[y.astype(int)],
                                       colorscale=[custom[0], custom[-1]],
                                       line=dict(color='black', width=1)),
                           showlegend=False)
    else:
        graph = go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                           marker=dict(color=y.astype(int),
                                       symbol=symbols[y.astype(int)],
                                       colorscale=[custom[0], custom[-1]],
                                       line=dict(color='black', width=1),
                                       size=weights),
                           showlegend=False)
    return surface, graph


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    train_X, train_y = generate_data(train_size, noise)
    test_X, test_y = generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    model = AdaBoost(DecisionStump, n_learners)
    model.fit(train_X, train_y)
    losses = []
    for T in range(1, n_learners + 1):
        train_partial_loss = model.partial_loss(train_X, train_y, T)
        test_partial_loss = model.partial_loss(test_X, test_y, T)
        losses.append((test_partial_loss, T, 'Test samples'))
        losses.append((train_partial_loss, T, 'Train samples'))
    df = pd.DataFrame(losses, columns=['Loss', 'Num of fitted learners',
                                       'Type'])
    px.line(df, x='Num of fitted learners', y='Loss', color='Type',
            title=f'Loss as function of number of base estimators used, noise: {noise}').show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    symbols = np.array(['', 'circle', 'square'])
    fig = make_subplots(rows=2, cols=2, horizontal_spacing=0.01, subplot_titles=[
        f"decision space using {t} models" for t in T])
    # performance = np.zeros((len(T), 2))
    for i, t in enumerate(T):
        # loss = model.partial_loss(test_X, test_y, t)
        # performance[i] = (t, loss)
        surface, graph = create_decision_surface_scatter(model, t, test_X,
                                                         test_y, symbols, lims)
        fig.add_trace(surface, row=int(i / 2) + 1, col=int(i % 2) + 1)
        fig.add_trace(graph, row=int(i / 2) + 1, col=int(i % 2) + 1)
    fig.update_layout(margin=dict(t=100)).update_xaxes(
        visible=False).update_yaxes(visible=False)
    fig.show()

    # Question 3: Decision surface of best performing ensemble
    # best_performing = int(performance[np.argmin(performance, axis=0)[1]][0])
    # prediction = model.partial_predict(test_X, best_performing)
    # acc = accuracy(test_y, prediction)
    size_to_loss = np.array([(i, model.partial_loss(test_X, test_y, i)) for i
                             in range(1, n_learners + 1)])
    best_t, best_loss = size_to_loss[np.argmin(size_to_loss, axis=0)[1]]
    best_t = int(best_t)
    prediction = model.partial_predict(test_X, best_t)
    best_accuracy = accuracy(test_y, prediction)
    acc2 = 1 - best_loss
    title = f"Best predicting model. Size: {best_t}, Accuracy: {best_accuracy}"
    surface, graph = create_decision_surface_scatter(model, best_t, test_X,
                                                     test_y, symbols, lims)
    fig2 = go.Figure()
    fig2.add_trace(surface)
    fig2.add_trace(graph)
    fig2.update_layout(title=title)
    fig2.show()

    # Question 4: Decision surface with weighted samples
    D = model.D_ / np.max(model.D_) * 5
    fig3 = go.Figure()
    surface, graph = create_decision_surface_scatter(model, n_learners,
                                                     train_X, train_y, symbols,
                                                     lims, D)
    fig3.add_trace(surface)
    fig3.add_trace(graph)
    fig3.update_layout(title=f"Samples with size based on probability, with noise: {noise}")
    fig3.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
