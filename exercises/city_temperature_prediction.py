from IMLearn.learners.regressors import PolynomialFitting
import IMLearn.learners.regressors.linear_regression
from IMLearn.utils import split_train_test
import plotly.express as px
import plotly.io as pio
import pandas as pd
import numpy as np

pio.templates.default = "simple_white"
DATA_PATH = '../datasets/City_Temperature.csv'
DATE_COL_INDEX = 2
DATE_COL = 'Date'
MONTH_COL = 'Month'
YEAR_COL = 'Year'
COUNTRY_COL = 'Country'
DAY_OF_YEAR_COL = 'DayOfYear'
CITY_COL = 'City'
DAY_COL = 'Day'
TEMP_COL = 'Temp'
COUNTRIES = ['Jordan', 'The Netherlands', 'South Africa']

def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=[DATE_COL_INDEX])
    df[DAY_OF_YEAR_COL] = df[DATE_COL].apply(lambda x: x.dayofyear)
    # month_dummies = pd.get_dummies(df[MONTH_COL])
    # df = pd.concat([df, month_dummies], axis=1)
    df.drop([DATE_COL, CITY_COL, DAY_COL], axis=1,
            inplace=True)
    df = df[(df[TEMP_COL] < 50) & (df[TEMP_COL] > -15)]
    return df

if __name__ == '__main__':
    np.random.seed(115)

    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data(DATA_PATH)

    # Question 2 - Exploring data for specific country
    israel_df = df[df[COUNTRY_COL] == "Israel"]
    # israel_df[YEAR_COL] = israel_df[YEAR_COL].astype(str)
    # px.scatter(israel_df, x=DAY_OF_YEAR_COL, y=TEMP_COL,
    #            color=israel_df[YEAR_COL].astype(str),
    #            title="Temperature by day of year").show()
    std_per_month = israel_df.groupby(MONTH_COL).agg(np.std)
    # px.bar(std_per_month, x=std_per_month.index, y=TEMP_COL,
    #        labels={TEMP_COL: 'Standard deviation of temperature'},
    #        title="Standard deviation of temperature each month").show()

    # Question 3 - Exploring differences between countries
    grouped_by = df.groupby([COUNTRY_COL, MONTH_COL]).agg({TEMP_COL: [np.mean, np.std]})
    grouped_by.columns = ['Temp mean', 'Temp std']
    grouped_by.reset_index(inplace=True)
    # px.line(grouped_by, x=MONTH_COL,
    #         y="Temp mean",
    #         color=COUNTRY_COL,
    #         error_y="Temp std").show()

    # Question 4 - Fitting model for different values of `k`
    israel_X_df = israel_df[[DAY_OF_YEAR_COL]]
    errors = []
    for k in range(1, 11):
        train_X, train_y, test_X, test_y = split_train_test(israel_X_df,
                                                            israel_df[TEMP_COL])
        model = PolynomialFitting(k)
        model.fit(train_X.to_numpy(), train_y.to_numpy())
        error = model.loss(test_X.to_numpy(), test_y.to_numpy())
        print(f"error for k={k}: {error}")
        errors.append((k, error))
    error_df = pd.DataFrame(errors, columns=['Polynomial degree', 'Error'])
    # px.bar(error_df, x="Polynomial degree", y='Error',
    #        title='Error in prediction by polynomial degree').show()

    # Question 5 - Evaluating fitted model on different countries
    model = PolynomialFitting(4)
    train_X, train_y = israel_X_df, israel_df[TEMP_COL]
    model.fit(train_X.to_numpy(), train_y.to_numpy())
    error_by_country = []
    for country in COUNTRIES:
        country_df = df[df[COUNTRY_COL] == country]
        test_X = country_df[[DAY_OF_YEAR_COL]]
        test_y = country_df[TEMP_COL]
        error = model.loss(test_X.to_numpy(), test_y.to_numpy())
        error_by_country.append((country, error))
    country_error_df = pd.DataFrame(error_by_country,
                                    columns=['Country', 'Error'])
    px.bar(country_error_df, x='Country', y='Error',
           title='Error in prediction by country').show()
