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
    is_train: bool
        Whether it is training set
    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    # Trying to use only interesting columns;

    # Originally used only:
    # 'original_selling_amount', 'is_user_logged_in', 'has_special_request',
    #    'days_in_advance', 'duration', 'guest_from_country',
    #    'percentage_of_payment_if_cancel_in_relevant_week',
    #    'amount_of_payment_if_cancel_in_relevant_week',
    #    'is_it_last_week_before_increase', 'has_no_show_policy'

    # Later tried using more: (but it didn't improve)
    # 'hotel_star_rating', 'guest_is_not_the_customer', 'no_of_adults',
    #    'no_of_children', 'no_of_extra_bed', 'no_of_room',
    #    'original_selling_amount', 'is_user_logged_in', 'request_nonesmoke',
    #    'request_latecheckin', 'request_highfloor', 'request_largebed',
    #    'request_twinbeds', 'request_airport', 'request_earlycheckin',
    #    'has_special_request', 'days_in_advance', 'duration',
    #    'guest_from_country',
    #    'percentage_of_payment_if_cancel_in_relevant_week',
    #    'amount_of_payment_if_cancel_in_relevant_week',
    #    'is_it_last_week_before_increase', 'has_no_show_policy'

    df = pd.read_csv(filename)
    date_columns = ["booking_datetime", "checkin_date", "checkout_date",
                    "cancellation_datetime", "hotel_live_date"]
    if not is_train:
        date_columns.remove("cancellation_datetime")
    for column in date_columns:
        df[column] = pd.to_datetime(df[column]).dt.floor("D")

    if is_train:
        # save only data where checkin is strictly after 15 of the next month,
        # in order to mimic test set
        df = df[df["booking_datetime"]
                + pd.to_timedelta((df["booking_datetime"].dt.days_in_month
                                   - df["booking_datetime"].dt.day + 15),
                                  unit="D")
                <= df["checkin_date"]]

    df["start_of_week_next_month"] = df["booking_datetime"] + pd.to_timedelta(
        (df["booking_datetime"].dt.days_in_month
         - df["booking_datetime"].dt.day + 7), unit="D")

    df["is_user_logged_in"] = df["is_user_logged_in"].astype(int)

    columns_to_drop = ["cancellation_datetime",
                       "h_booking_id", "booking_datetime",
                       "checkin_date", "checkout_date",
                       "hotel_id", "hotel_country_code", "hotel_live_date",
                       "accommadation_type_name",
                       "h_customer_id",
                       "customer_nationality",
                       "guest_nationality_country_name",
                       "origin_country_code",
                       "language", "original_payment_method",
                       "original_payment_type", "original_payment_currency",
                       "cancellation_policy_code",
                       "hotel_area_code", "hotel_brand_code",
                       "hotel_city_code", "hotel_chain_code",
                       "start_of_week_next_month",
                       "hotel_star_rating", "no_of_room",
                       "guest_is_not_the_customer",
                       "no_of_adults", "no_of_children", "no_of_extra_bed",
                       # Columns I tried to return:
                       "is_first_booking",
                       "charge_option",
                       "cancelled_in_these_dates"]

    df["has_special_request"] = 0
    special_requests_prefix = "request_"
    for column in df.columns:
        if column.startswith(special_requests_prefix):
            df[column] = df[column].fillna(0)
            df["has_special_request"] = df["has_special_request"] + df[column]
            # Columns I tried to return:
            columns_to_drop.append(column)
    df["has_special_request"] = df["has_special_request"].apply(
        lambda x: int(x > 0))

    df["days_in_advance"] = (df["checkin_date"]
                             - df["booking_datetime"]).dt.days
    df["duration"] = (df["checkout_date"] - df["checkin_date"]).dt.days

    df["guest_from_country"] = df["hotel_country_code"] == df[
                                                        "origin_country_code"]
    df["guest_from_country"] = df["guest_from_country"].astype(int)

    # Columns I tried to return:
    # charge_option_dict = {"Pay Now": 0, "Pay Later": 1, "Pay at Check-in": 2}
    # df["charge_option"] = df["charge_option"].apply(lambda x:
    #                                                 charge_option_dict.get(x,
    #                                                                       0))

    # common_accommodation_type = ["Hotel", "Resort",
    #                              "Guest House / Bed & Breakfast",
    #                              "Hostel", "Serviced Apartment", "Apartment"]
    # df.accommadation_type_name = df.accommadation_type_name.apply(
    #     lambda x: x if x in common_accommodation_type else "other")
    # dummies = pd.get_dummies(df.accommadation_type_name)
    # for accom_type in common_accommodation_type:
    #     if accom_type not in dummies.columns:
    #         dummies[accom_type] = 0
    # dummies = dummies[common_accommodation_type]
    # df = pd.concat([df, dummies], axis=1)

    if is_train:
        df = df[df.cancellation_policy_code != "UNKNOWN"]

    df["percentage_of_payment_if_cancel_in_relevant_week"] = 0
    df["amount_of_payment_if_cancel_in_relevant_week"] = 0
    df["is_it_last_week_before_increase"] = 0
    has_noshow_policy = []
    cancelled_in_these_dates = []
    for i, row in df.iterrows():
        days_to_cancel, has_noshow, all_policies = _parse_cancellation_code(
            row['cancellation_policy_code'])
        descending_keys = list(all_policies)
        descending_keys.sort(reverse=True)
        relevant_policy_days = None
        if all_policies:
            for num_of_days in descending_keys:
                relevant_day_until_checkin = df.loc[i, "checkin_date"] \
                        - pd.to_timedelta(num_of_days, unit="D")
                if relevant_day_until_checkin \
                        >= df.loc[i, "start_of_week_next_month"]:
                    relevant_policy_days = num_of_days
                if df.loc[i, "start_of_week_next_month"] \
                        + pd.to_timedelta(8, unit="D") \
                        >= relevant_day_until_checkin \
                        > df.loc[i, "start_of_week_next_month"]:
                    df.loc[i, "is_it_last_week_before_increase"] = 1
        if relevant_policy_days:
            if all_policies[relevant_policy_days][1] == "P":
                df.loc[i, "percentage_of_payment_if_cancel_in_relevant_week"] \
                    = all_policies[relevant_policy_days][0] / 100
                df.loc[i, "amount_of_payment_if_cancel_in_relevant_week"] \
                    = all_policies[relevant_policy_days][0] / 100 * \
                      df.loc[i, "original_selling_amount"]
            else: # it is "N"
                df.loc[i, "percentage_of_payment_if_cancel_in_relevant_week"] \
                    = all_policies[relevant_policy_days][0] / \
                      df.loc[i, "duration"]
                df.loc[i, "amount_of_payment_if_cancel_in_relevant_week"] \
                    = all_policies[relevant_policy_days][0] / \
                      df.loc[i, "duration"] * \
                      df.loc[i, "original_selling_amount"]

        has_noshow_policy.append(int(has_noshow))
        if is_train:
            cancellation_time = row["cancellation_datetime"]
            if not isinstance(cancellation_time, datetime.date) or pd.isna(
                    cancellation_time):
                cancelled_in_these_dates.append(0)
            else:
                was_canceled = row["start_of_week_next_month"] <= \
                               row["cancellation_datetime"] < \
                               row["start_of_week_next_month"] + \
                               pd.to_timedelta(7, unit="D")
                cancelled_in_these_dates.append(int(was_canceled))
    df["has_no_show_policy"] = has_noshow_policy
    if is_train:
        df["cancelled_in_these_dates"] = cancelled_in_these_dates
    else:
        df['cancelled_in_these_dates'] = 0

    results = df["cancelled_in_these_dates"]

    if not is_train:
        columns_to_drop.remove("cancellation_datetime")
    df.drop(columns_to_drop, axis=1, inplace=True)

    return df, results


def _parse_cancellation_code(code: str) -> Tuple[int, bool, dict]:
    """
    parses cancellation code to days to cancel and whether there is a
    no show policy.
    addition: a dict of all cancellation codes
    """
    if not code or not isinstance(code, str):
        return 0, False, dict()
    if code == "UNKNOWN":
        # # Original:
        # return 5, True
        # # My change:
        return 0, True, dict()
    parts = code.split("_")
    has_no_show_policy = False
    days_in_advance = 0
    policy_by_days = dict()
    for part in parts:
        if "D" not in part:
            has_no_show_policy = True
            continue
        else:
            num_of_days = int(part[:part.find("D")])
            days_in_advance = max(days_in_advance, num_of_days)
            policy_by_days[num_of_days] = (int(part[part.find("D") + 1:-1]),
                                           part[-1])
    return days_in_advance, has_no_show_policy, policy_by_days


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
    pd.DataFrame(estimator.predict_with_threshold(X, threshold=0.08),
                 columns=["predicted_values"]).to_csv(filename, index=False)


if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    X, y = load_data("../datasets/agoda_cancellation_train.csv", is_train=True)

    # test, _ = load_data("./test_sets/week_7_test_data.csv", is_train=False)

    week1_X, _ = load_data("./test_sets/test_set_week_1.csv", is_train=False)
    week1_y = pd.read_csv("./test_sets/week_1_labels.csv")["cancel"]
    week2_X, _ = load_data("./test_sets/test_set_week_2.csv", is_train=False)
    week2_y = pd.read_csv("./test_sets/week_2_labels.csv")["cancel"]
    week3_X, _ = load_data("./test_sets/test_set_week_3.csv", is_train=False)
    week3_y = pd.read_csv("./test_sets/week_3_labels.csv")["cancel"]
    week4_X, _ = load_data("./test_sets/test_set_week_4.csv", is_train=False)
    week4_y = pd.read_csv("./test_sets/week_4_labels.csv")["cancel"]
    week5_X, _ = load_data("./test_sets/week_5_test_data.csv", is_train=False)
    week5_y = pd.read_csv("./test_sets/week_5_labels.csv")["cancel"]
    week6_X, _ = load_data("./test_sets/week_6_test_data.csv", is_train=False)
    week6_y = pd.read_csv("./test_sets/week_6_labels.csv")["cancel"]

    # train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25)

    # print("\nWeek 1 loss test")
    # check_estimator_on_labels1 = AgodaCancellationEstimator()
    # check_estimator_on_labels1.fit(X, y)
    # check_estimator_on_labels1.loss(week1_X, week1_y)
    # print("\nWeek 2 loss test")
    # check_estimator_on_labels2 = AgodaCancellationEstimator()
    # check_estimator_on_labels2.fit(X, y)
    # check_estimator_on_labels2.loss(week2_X, week2_y)
    # print("\nWeek 3 loss test")
    # check_estimator_on_labels3 = AgodaCancellationEstimator()
    # check_estimator_on_labels3.fit(X, y)
    # check_estimator_on_labels3.loss(week3_X, week3_y)
    # print("\nWeek 4 loss test")
    # check_estimator_on_labels4 = AgodaCancellationEstimator()
    # check_estimator_on_labels4.fit(X, y)
    # check_estimator_on_labels4.loss(week4_X, week4_y)
    # print("\nWeek 5 loss test")
    # check_estimator_on_labels1 = AgodaCancellationEstimator()
    # check_estimator_on_labels1.fit(X, y)
    # check_estimator_on_labels1.loss(week5_X, week5_y)
    # print("\nWeek 6 loss test")
    # check_estimator_on_labels1 = AgodaCancellationEstimator()
    # check_estimator_on_labels1.fit(X, y)
    # check_estimator_on_labels1.loss(week6_X, week6_y)
    #
    # print("\nTrain-Test partition loss test")
    # check_estimator_on_train = AgodaCancellationEstimator()
    # check_estimator_on_train.fit(train_X, train_y)
    # check_estimator_on_train.loss(test_X.to_numpy(), test_y.to_numpy())

    # Test for checking multiple estimators
    for i in range(2, 11):
        print(f"Starting weight {i}")
        samples = [
            (week1_X, week1_y),
            (week2_X, week2_y),
            (week3_X, week3_y),
            (week4_X, week4_y),
            (week5_X, week5_y),
            (week6_X, week6_y),
        ]
        weights = np.ones(y.shape[0])
        for j, val in enumerate(y):
            if val == 1:
                weights[j] = i
        estimator = AgodaCancellationEstimator(single=False)
        estimator.fit_with_weight(X, y, weights)
        df = estimator.loss_multiple(samples)
        df.to_csv(f'./results/comparison_weight_{i}.csv', index=False)
        print(f"Finished weight {i}")

    # results = np.zeros(6)
    # for i, sample in enumerate(samples):
    #     test_X, test_y = sample
    #     results[i] = estimator.loss(test_X, test_y)
    # mean = np.mean(results)
    # median = np.median(results)
    # min_res = np.min(results)
    # max_res = np.max(results)
    # print(f"regular estimator: mean {mean}, median {median}, min {min_res}, max {max_res}")

    # Fit model over data
    # estimator = AgodaCancellationEstimator()
    # estimator.fit(X, y)
    #
    # # Store model predictions over test set
    # evaluate_and_export(estimator, test.to_numpy(),
    #                     "2055501016_208543116_207129420.csv")