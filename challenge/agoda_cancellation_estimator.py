
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
from sklearn.neural_network import MLPRegressor, MLPClassifier


def calc_f1_macro(y_true, y_pred):
    """
    @param y_true: The true values
    @param y_pred: The predicted values
    @return the f1 macro results for true and predicted values
    """
    tp1 = tn1 = fp1 = fn1 = 0
    tp0 = tn0 = fp0 = fn0 = 0
    for i, val in enumerate(y_pred):
        if y_true[i] == val:
            if val == 1:
                tp1 += 1
                tn0 += 1
            else:
                tn1 += 1
                tp0 += 1
        else:
            if val == 1:
                fp1 += 1
                fn0 += 1
            else:
                fn1 += 1
                fp0 += 1
    f1_1 = tp1 / (tp1 + 0.5 * (fp1 + fn1))
    f1_0 = tp0 / (tp0 + 0.5 * (fp0 + fn0))
    f1_macro = (f1_0 + f1_1) * 0.5
    return f1_macro

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
        ## Our chosen non-weighted: (later we returned its weights
        # self.model = RandomForestRegressor(max_depth=3, random_state=0)
        self.model = MLPRegressor(hidden_layer_sizes=(10, 8, 6, 4), solver='adam',
                                  random_state=17, max_iter=500)
        ## After running many combinations of hyperparameters:
        # self.model = RandomForestRegressor(max_depth=3, random_state=0,
        #                                    n_estimators=120)
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
        # self.model = RandomForestRegressor(max_depth=3, random_state=0)
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
            self.models = []
            for model_type, model_name in [(MLPRegressor, "MLPRegressor")]:
                for solver in ['lbfgs']:
                    for hidden_layers_structure in [(60, 60)]:
                        desc = f"{model_name}, hidden layers: {hidden_layers_structure}, " \
                               f"solver: {solver}"
                        model = model_type(hidden_layer_sizes=hidden_layers_structure,
                                           solver=solver, random_state=17)
                        self.models.append((model, desc))

            ### Original tests:
            # self.models = []
            # # for i in range(1, 6):
            # for i in range(2, 4):
            #     # for j in range(50, 151, 10):
            #     for j in range(100, 151, 10):
            #         desc = f"Random forest - Depth {i}, estimators: {j}"
            #         model = RandomForestRegressor(max_depth=i, random_state=0,
            #                                       n_estimators=j)
            #         self.models.append((model, desc))
            #     # desc = f"Decision tree - Depth {i}"
                # DecisionTreeRegressor(max_depth=i)
                # self.models.append((model, desc))
#            for i in range(10, 21):
#                desc = f"Adaboost regressor - estimators: {i}"
#                model = AdaBoostRegressor(n_estimators=i, random_state=0)
#                self.models.append((model, desc))
#            for i in range(3, 20):
#                desc = f"Knn with {i} neighbors"
#                model = KNeighborsRegressor(n_neighbors=i)
#                self.models.append((model, desc))
        else:
            self.models = None

    def fit_with_weight(self, X: np.ndarray, y: np.ndarray,
                        weights: np.ndarray) -> NoReturn:
        self.model.fit(X, y, sample_weight=weights)
        if self.models is not None:
            for model, _ in self.models:
                model.fit(X, y, sample_weight=weights)
        self.fitted_ = True

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

    def predict_with_threshold(self, X: np.ndarray,
                               # threshold: float = 0.08) \
                               # threshold: float = 0.154924874791318) \
                               threshold: float = 0.155) \
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
        # threshold_options = [i / 100 for i in range(1, 11)]
        threshold_options = [0.395, 0.415, 0.45]
        # threshold_options = [0.08]
        for threshold in threshold_options:
            res = self.predict_with_threshold(X, threshold)
            total = res.sum()
            f1_macro = calc_f1_macro(y, res)
            f1_macros.append(f1_macro)
            print(f"threshold: {threshold}, f1 macro: {f1_macro}, total cancellations: {total}")
        return max(f1_macros)

    def loss_multiple(self, samples: List[Tuple[np.ndarray, np.ndarray]]) -> \
            pd.Dataframe:
        if self.models is None:
            return
        results = np.zeros((len(self.models), 5))
        descriptions = []
        thresholds = np.linspace(0, 0.6, 800)
        # thresholds = np.linspace(0, 0.4, 600)
        # thresholds = [0.08, 0.1, 0.4]
        # thresholds = [i / 100 for i in range(1, 11)]
        # thresholds = [i / 10 for i in range(1, 10)]
        for i, m in enumerate(self.models):
            model, desc = m
            mean, median, min_pred, max_pred, best_threshold = 0, 0, 0, 0, 0
            descriptions.append(desc)
            model_predictions = [model.predict(X) for X, _ in samples]
            for threshold in thresholds:
                predictions = []
                for j, data in enumerate(samples):
                    _, y = data
                    y_pred = apply_threshold(model_predictions[j], threshold)
                    f1_macro = calc_f1_macro(y, y_pred)
                    predictions.append(f1_macro)
                predictions = np.array(predictions)
                current_min = np.min(predictions)
                if current_min > min_pred:
                    median = np.median(predictions)
                    mean = np.mean(predictions)
                    min_pred = np.min(predictions)
                    max_pred = np.max(predictions)
                    best_threshold = threshold
            results[i] = (best_threshold, median, mean, min_pred, max_pred)
            print(f"Finished going over {desc}")
        df = pd.DataFrame(results, columns=['threshold', 'median', 'mean',
                                            'min', 'max'])
        df['description'] = descriptions
        return df