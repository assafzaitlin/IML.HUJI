from __future__ import annotations
from typing import NoReturn, List, Tuple
from IMLearn.base import BaseEstimator
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, \
    RandomForestRegressor, BaggingRegressor, AdaBoostRegressor, \
    GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, \
    QuadraticDiscriminantAnalysis


def f1(y_true, y_pred):
    tp1 = tn1 = fp1 = fn1 = 0
    tp0 = tn0 = fp0 = fn0 = 0
    for i, prediction in enumerate(y_pred):
        if y_true[i] == prediction:
            if i == 1:
                tp1 += 1
                tn0 += 1
            else:
                tn1 += 1
                tp0 += 1
        else:
            if i == 1:
                fp1 += 1
                fn0 += 1
            else:
                fn1 += 1
                fp0 += 1
    f1_1 = tp1 / (tp1 + 0.5 * (fp1 + fn1))
    f1_0 = tp0 / (tp0 + 0.5 * (fp0 + fn0))
    return (f1_0 + f1_1) * 0.5

def apply_threshold(prediction, threshold):
    """
    Applies threshold on prediction
    """
    return np.array([1 if i >= threshold else 0 for i in prediction])

class AgodaCancellationEstimator(BaseEstimator):
    """
    An estimator for solving the Agoda Cancellation challenge
    """

    def __init__(self, single=True) -> AgodaCancellationEstimator:
        """
        Instantiate an estimator for solving the Agoda Cancellation challenge
        Parameters
        ----------
        Attributes
        ----------
        """
        super().__init__()
        ## NEW:
        # self.model = AdaBoostClassifier(n_estimators=100)
        # self.model = AdaBoostClassifier(n_estimators=100, random_state=0)
        # self.model = AdaBoostRegressor(n_estimators=100, random_state=0)
        # self.model = AdaBoostRegressor(n_estimators=150, random_state=0)
        # self.model = RandomForestClassifier(max_depth=2, random_state=0)
        # self.model = RandomForestClassifier(max_depth=5, random_state=0)
        # self.model = RandomForestClassifier(max_depth=5, random_state=0)
        # self.model = RandomForestClassifier(max_depth=10, random_state=0)
        # self.model = RandomForestRegressor(max_depth=2, random_state=0)
        self.model = RandomForestRegressor(max_depth=3, random_state=0)
        # self.model = RandomForestRegressor(max_depth=4, random_state=0)
        # self.model = RandomForestRegressor(max_depth=5, random_state=0)
        # self.model = RandomForestRegressor(max_depth=10, random_state=0)
        # self.model = BaggingRegressor()
        # self.model = BaggingRegressor(base_estimator=SVR())
        # self.model = BaggingRegressor(base_estimator=LinearRegression())
        # self.model = BaggingRegressor(base_estimator=QuadraticDiscriminantAnalysis())
        # self.model = GradientBoostingRegressor(random_state=0)
        # # Original:
        # Over week1 and week2, LinearRegression is best with threshold = 0.08
        # self.model = LinearRegression()
        # # New tries:
        # self.model = LogisticRegression() # 10
        # self.model = DecisionTreeClassifier(max_depth=2) # 11
        # self.model = DecisionTreeClassifier(max_depth=5) # 12
        # self.model = DecisionTreeRegressor(max_depth=2)  # 12
        # self.model = DecisionTreeRegressor(max_depth=5)  # 12
        # self.model = KNeighborsClassifier(n_neighbors=5) # 13
        # self.model = KNeighborsClassifier(n_neighbors=10) # 14
        # self.model = KNeighborsClassifier(n_neighbors=16) # 14
        # self.model = KNeighborsRegressor(n_neighbors=5) # 14
        # self.model = KNeighborsRegressor(n_neighbors=10) # 14
        # self.model = KNeighborsRegressor(n_neighbors=15) # 14
        # self.model = make_pipeline(PolynomialFeatures(2),
        #                            LinearRegression(fit_intercept=False)) #15
        # Over train-test split, polynomial is best (k=3 with threshold=0.25)
        # self.model = make_pipeline(PolynomialFeatures(3),
        #                            LinearRegression(fit_intercept=False)) #16
        # self.model = make_pipeline(PolynomialFeatures(5),
        #                            LinearRegression(fit_intercept=False)) #17
        # self.model = SVC(gamma="auto") #18
        # self.model = LinearDiscriminantAnalysis(store_covariance=True) #19
        # self.model = QuadraticDiscriminantAnalysis(store_covariance=True) #20

        if not single:
            self.models = [(LogisticRegression(), 'Logistic regression')]
            for i in range(1, 11):
                desc = f"Random forest - Depth {i}"
                model = RandomForestRegressor(max_depth=i, random_state=0)
                self.models.append((model, desc))
                desc = f"Decision tree - Depth {i}"
                DecisionTreeRegressor(max_depth=i)
                self.models.append((model, desc))
            for i in range(80, 121):
                desc = f"Adaboost regressor - estimators: {i}"
                model = AdaBoostRegressor(n_estimators=i, random_state=0)
                self.models.append((model, desc))
            for i in range(3, 20):
                desc = f"Knn with {i} neighbors"
                model = KNeighborsRegressor(n_neighbors=i)
                self.models.append((model, desc))
        else:
            self.models = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an estimator for given samples
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for
        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        Notes
        -----
        """
        self.model.fit(X, y)
        if self.models is not None:
            for model, _ in self.models:
                model.fit(X, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        this is a fictive function, only for the API to run
        """
        return self.predict_with_threshold(X)

    def predict_with_threshold(self, X: np.ndarray, threshold: float = 0.08) \
            -> np.ndarray:
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
        return np.array([1 if i >= threshold else 0
                         for i in self.model.predict(X)])

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under loss function
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples
        y : ndarray of shape (n_samples, )
            True labels of test samples
        Returns
        -------
        loss : float
            Performance under loss function
        """
        f1_macros = []
        # threshold_options = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9]
        threshold_options = [i / 100 for i in range(1, 11)]
        # threshold_options = [i / 100 + 0.1 for i in range(1, 11)]
        threshold_options = [0.08]
        for threshold in threshold_options:
            res = self.predict_with_threshold(X, threshold)
            tp1 = tn1 = fp1 = fn1 = 0
            tp0 = tn0 = fp0 = fn0 = 0
            for index, i in enumerate(res):
                if y[index] == i:
                    if i == 1:
                        tp1 += 1
                        tn0 += 1
                    else:
                        tn1 += 1
                        tp0 += 1
                else:
                    if i == 1:
                        fp1 += 1
                        fn0 += 1
                    else:
                        fn1 += 1
                        fp0 += 1
            f1_1 = tp1 / (tp1 + 0.5 * (fp1 + fn1))
            f1_0 = tp0 / (tp0 + 0.5 * (fp0 + fn0))
            f1_macro = (f1_0 + f1_1) * 0.5
            f1_macros.append(f1_macro)
            accuracy = (tp1 + tn1) / len(res)
            # print(f"threshold: {threshold}, f1 for 1s: {f1_1},"
            #       f" f1 for 0s: {f1_0}, , f1 macro: {f1_macro}, "
            #       f"accuracy: {accuracy}")
            print(f"threshold: {threshold}, f1 macro: {f1_macro}")
        return max(f1_macros)

    def loss_multiple(self, samples: List[Tuple[np.ndarray, np.ndarray]]) -> \
            pd.Dataframe:
        if self.models is None:
            return
        results = np.zeros((len(self.models), 5))
        descriptions = []
        thresholds = np.linspace(0.01, 0.5, 200)
        for i, m in enumerate(self.models):
            model, desc = m
            best_avg = 0
            descriptions.append(desc)
            for threshold in thresholds:
                predictions = []
                for X, y in samples:
                    y_pred = apply_threshold(model.predict(X), threshold)
                    f1_macro = f1(y, y_pred)
                    predictions.append(f1_macro)
                predictions = np.array(predictions)
                avg = np.average(predictions)
                if avg > best_avg:
                    median = np.median(predictions)
                    best_avg = avg
                    min_pred = np.min(predictions)
                    max_pred = np.max(predictions)
                    results[i] = (threshold, median, avg, min_pred, max_pred)
            print(f"Finished going over {desc}")
        df = pd.DataFrame(results, columns=['threshold', 'median', 'avg',
                                            'min', 'max'])
        df['description'] = descriptions
        return df
