from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

import numpy as np
import pandas as pd
import datetime


def load_data(filename: str, is_train: bool = False):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset
    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    df = pd.read_csv(filename).drop_duplicates()
    special_requests_prefix = "request_"
    for column in df.columns:
        if column.startswith(special_requests_prefix):
            df[column] = df[column].fillna(0)
    if is_train:
        df = df[df.cancellation_policy_code != "UNKNOWN"]
    date_columns = ["booking_datetime", "checkin_date", "checkout_date",
                    "cancellation_datetime", "hotel_live_date"]
    if not is_train:
        date_columns.remove("cancellation_datetime")
    for column in date_columns:
        df[column] = pd.to_datetime(df[column])
        df[column] = df[column].apply(lambda x: x.date() if x else x)
    common_accommodation_type = ["Hotel", "Resort",
                                "Guest House / Bed & Breakfast",
                                "Hostel", "Serviced Apartment", "Apartment"]
    df["days_in_advance"] = df["checkin_date"] - df["booking_datetime"]
    df.days_in_advance = df.days_in_advance.apply(lambda x: x.days)
    #df["duration"] = df["checkout_date"] - df["checkin_date"]
    #df.duration = df.duration.apply(lambda x: x.days)
    #df["guest_from_country"] = df["hotel_country_code"] == df["origin_country_code"]
    #df.guest_from_country = df.guest_from_country.astype(int)
    # df["is_new_hotel"] = df["booking_datetime"] - df["hotel_live_date"]
    # df.is_new_hotel = df.is_new_hotel.apply(lambda x: x.days)
    # df.is_new_hotel = df.is_new_hotel.apply(lambda x: int(x < 180))
    df.accommadation_type_name = df.accommadation_type_name.apply(
        lambda x: x if x in common_accommodation_type else "other")
    dummies = pd.get_dummies(df.accommadation_type_name)
    for accom_type in common_accommodation_type:
        if accom_type not in dummies.columns:
            dummies[accom_type] = 0
    dummies = dummies[common_accommodation_type]
    df = pd.concat([df, dummies], axis=1)
    # charge_option_dict = {"Pay Now": 0, "Pay Later": 1, "Pay at Check-in": 2}
    # df.charge_option = df.charge_option.apply(lambda x: charge_option_dict.get(x, 0))
    # df["large_group"] = df["no_of_adults"].apply(lambda x: int(x > 4))
    # df["has_children"] = df["no_of_children"].apply(lambda x: int(x > 0))
    # df["has_extra_beds"] = df["no_of_extra_bed"].apply(lambda x: int(x > 0))
    df.is_user_logged_in = df.is_user_logged_in.astype(int)
    # df["has_special_request"] = 0
    # for column in df.columns:
    #     if column.startswith(special_requests_prefix):
    #         df["has_special_request"] = df["has_special_request"] + df[column]
    # df["has_special_request"] = df["has_special_request"].apply(lambda x: int(x > 0))
    has_noshow_policy = []
    can_cancel_within_4_weeks = []
    cancelled = []
    for i, row in df.iterrows():
        days_to_cancel, has_noshow = _parse_cancellation_code(row['cancellation_policy_code'])
        can_cancel = int(row['days_in_advance'] - days_to_cancel > 28)
        has_noshow_policy.append(has_noshow)
        can_cancel_within_4_weeks.append(can_cancel)
        if is_train:
            cancellation_time = row["cancellation_datetime"]
            if not isinstance(cancellation_time, datetime.date) or pd.isna(cancellation_time):
                cancelled.append(0)
            else:
                diff = cancellation_time - row["booking_datetime"]
                cancelled.append(int(diff.days <= 28))
    df["has_no_show_policy"] = has_noshow_policy
    df["can_cancel_4_weeks"] = can_cancel_within_4_weeks
    if is_train:
        df["cancelled"] = cancelled
    else:
        df['cancelled'] = 0
    df = df[df.days_in_advance >= 0]
    results = df["cancelled"]
    # TODO remember to remove charge_option if we bring it back
    columns_to_drop = ["h_booking_id", "booking_datetime",
             "checkin_date", "checkout_date",
             "cancellation_datetime", "hotel_id", "hotel_country_code",
             "origin_country_code", "hotel_live_date",
             "accommadation_type_name", "h_customer_id",
             "customer_nationality", "guest_nationality_country_name",
             "no_of_adults", "no_of_children", "no_of_extra_bed",
             "language", "original_payment_method", "original_payment_type",
             "original_payment_currency", "hotel_area_code",
             "hotel_city_code", "hotel_chain_code", "hotel_brand_code",
                       "cancellation_policy_code",
             "cancelled", "charge_option", "is_first_booking"]
    if not is_train:
        columns_to_drop.remove("cancellation_datetime")
    df.drop(columns_to_drop,
            axis=1, inplace=True)
    return df, results


def _parse_cancellation_code(code: str) -> Tuple[int, bool]:
    """
    parses cancellation code to days to cancel and whether or not there is a
    no show policy.
    """
    if not code or not isinstance(code, str):
        return 0, False
    if code == "UNKNOWN":
        return 5, True
    parts = code.split("_")
    has_no_show_policy = False
    days_in_advance = 0
    for part in parts:
        if "D" not in part:
            has_no_show_policy = True
            continue
        else:
            num_of_days = int(part[:part.find("D")])
            days_in_advance = max(days_in_advance, num_of_days)
    return days_in_advance, has_no_show_policy

def evaluate_and_export(estimator, X: np.ndarray, filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.
    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.
    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction
    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses
    filename:
        path to store file at
    """
    pd.DataFrame(estimator.predict(X),
                 columns=["predicted_values"]).to_csv(filename, index=False)


if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    X, y = load_data("../datasets/agoda_cancellation_train.csv", is_train=True)
    test, _ = load_data("./test_sets/test_set_week_2.csv", is_train=False)

    #train_X, test_X, train_y, test_y = train_test_split(df, cancellation_labels, test_size=0.25)

    # Fit model over data
    estimator = AgodaCancellationEstimator().fit(X.to_numpy(), y.to_numpy())

    # Store model predictions over test set
    evaluate_and_export(estimator, test.to_numpy(),
                        "results/2055501016_208543116_207129420.csv")