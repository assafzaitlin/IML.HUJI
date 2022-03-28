from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from dateutil import parser
pio.templates.default = "simple_white"
DATE_COL = 'date'
BEDROOMS_COL = 'bedrooms'
BATHROOMS_COL = 'bathrooms'
LOT_AREA_COL = 'sqft_lot'
HOUSE_AREA_COL = 'sqft_living'
FLOORS_COL = 'floors'
BUILT_COL = 'yr_built'
RENOVATED_COL = 'yr_renovated'
ZIPCODE_COL = 'zipcode'
LOT_AREA15_COL = 'sqft_lot15'
HOUSE_AREA15_COL = 'sqft_living15'
CONDITION_COL = 'condition'
WATERFRONT_COL = 'waterfront'
VIEW_COL = 'view'
GRADE_COL = 'grade'
HAS_YARD_COL = 'has_yard'
GRADE_RANGE = list(range(1, 14))
VIEW_RANGE = list(range(5))
CONDITION_RANGE = list(range(1, 6))
DF_COLUMNS = [DATE_COL, BEDROOMS_COL, BATHROOMS_COL, HOUSE_AREA_COL,
              LOT_AREA_COL, FLOORS_COL, WATERFRONT_COL, VIEW_COL,
              CONDITION_COL, GRADE_COL, 'sqft_above', 'sqft_basement',
              BUILT_COL, RENOVATED_COL, ZIPCODE_COL, HOUSE_AREA15_COL,
              LOT_AREA15_COL]
PRICE_COL = 'price'
AGE_COL = 'house_age'
YARD_COL = 'sqft_yard'
YARD15_COL = 'sqft_yard15'
LAST_RENOVATED_COL = 'years_since_renovated'
MINIMAL_YEAR = 2000 # Sales before that year are probably invalid data
TRAIN_DATA_PATH = '../datasets/house_prices.csv'
OTHER_ZIPCODE = 'zipcode_other'
FEATURES_TO_SAVE = [HOUSE_AREA_COL, AGE_COL]

def _get_year_from_date(date_str: str) -> int:
    """
    @param date_str: The string representing the date
    Returns the sum of year * 12 + month of the date
    """
    if not isinstance(date_str, str) or date_str == '0':
        return 0
    try:
        parsed = parser.parse(date_str)
        return parsed.year
    except Exception as e:
        return 0

def load_data(filename: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices)
    """
    df = pd.read_csv(filename)
    df.fillna(0, inplace=True)
    df = df[DF_COLUMNS + [PRICE_COL]]
    df[DATE_COL] = df[DATE_COL].apply(_get_year_from_date)
    # Removes rows with irrelevant values
    df = df[(df[DATE_COL] > MINIMAL_YEAR) & (df[PRICE_COL] > 0) &
            (df[BEDROOMS_COL] > 0) & (df[FLOORS_COL] > 0) &
            (df[BATHROOMS_COL] > 0) & (df[HOUSE_AREA_COL] > 0) &
            (df[LOT_AREA_COL] > 0) & (df[BUILT_COL] > 0) &
            (df[WATERFRONT_COL].isin([0, 1])) &
            (df[GRADE_COL].isin(GRADE_RANGE)) &
            (df[VIEW_COL].isin(VIEW_RANGE)) &
            (df[CONDITION_COL].isin(CONDITION_RANGE))]
    df[YARD_COL] = df[LOT_AREA_COL] - df[HOUSE_AREA_COL]
    df[HAS_YARD_COL] = (df[YARD_COL] > 0).apply(int)
    df[YARD15_COL] = df[LOT_AREA15_COL] - df[HOUSE_AREA15_COL]
    df[RENOVATED_COL] = df[[RENOVATED_COL, BUILT_COL]].max(axis=1)
    df[AGE_COL] = df[DATE_COL] - df[BUILT_COL]
    df[ZIPCODE_COL] = df[ZIPCODE_COL].apply(lambda x: f'zipcode_{x}')
    zipcode_dummies = pd.get_dummies(df[ZIPCODE_COL])
    df = pd.concat([df, zipcode_dummies], axis=1)
    df = df[(df[AGE_COL] >= 0) & (df[YARD_COL] >= 0) &
            (df[YARD15_COL] >= 0)]
    results = df[PRICE_COL]
    df.drop([ZIPCODE_COL, PRICE_COL, DATE_COL, BUILT_COL,
             LOT_AREA_COL, LOT_AREA15_COL], axis=1, inplace=True)
    return df, results


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    y_std = y.std()
    for feature_name in X.columns:
        feature = X[feature_name]
        cov = feature.cov(y)
        denominator = feature.std() * y_std
        pearson_correlation = cov / denominator
        if feature_name in FEATURES_TO_SAVE:
            df = pd.DataFrame(columns=['feature', 'result'])
            df.feature = feature
            df.result = y
            title = f'{feature_name}_{pearson_correlation}'
            fig = px.scatter(df, x='feature', y='result',
                             title=title)
            path = f'{output_path}/{feature_name}.jpg'
            fig.write_image(path)


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data(TRAIN_DATA_PATH)

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y)

    # Question 3 - Split samples into training- and testing sets.
    training_X, training_y, test_X, test_y = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    X = test_X.to_numpy()
    y = test_y.to_numpy()
    losses = []
    for p in range(10, 101):
        loss_over_same_pctg = []
        pctg = p / 100
        for i in range(10):
            partial_training_X = training_X.sample(frac=pctg)
            partial_training_y = training_y.take(partial_training_X.index)
            model = LinearRegression()
            model.fit(partial_training_X.to_numpy(),
                      partial_training_y.to_numpy())
            loss = model.loss(X, y)
            loss_over_same_pctg.append(loss)
        loss_mean = np.mean(loss_over_same_pctg)
        loss_std = np.std(loss_over_same_pctg)
        losses.append((p, loss_mean, loss_std))
    mean = np.array([i[1] for i in losses])
    std = np.array([i[2] for i in losses])
    pctg = list(range(10, 101))
    f = go.Scatter(x=pctg, y=mean, mode='markers+lines', name='Mean loss',
                   marker={'color': 'blue', 'opacity': .7})
    f2 = go.Scatter(x=pctg, y=mean - 2 * std, fill=None, mode='lines',
                    line={'color': 'lightgrey'}, showlegend=False)
    f3 = go.Scatter(x=pctg, y=mean + 2 * std, fill='tonexty', mode='lines',
                    line={'color': 'lightgrey'}, showlegend=False)
    layout = go.Layout(title="Mean loss of houses' price prediction by learning set size",
                       xaxis={'title': '% of learning set size'},
                       yaxis={'title': 'Mean loss'})
    frame = go.Frame(data=[f, f2, f3], layout=layout)
    fig = go.Figure(data=frame['data'], frames=[frame],
                    layout=go.Layout(title=frame['layout']['title'],
                                     xaxis=frame['layout']['xaxis'],
                                     yaxis=frame['layout']['yaxis'])
                    )
    fig.show()
