import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.base import TransformerMixin, BaseEstimator

DEFAULT_PARAMS = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "metric": "auc",
    "max_depth": 4,
    "learning_rate": 0.01,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "n_estimators": 1000,
    "random_state": 42,
}


class FuzzyAugmentation(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        clf=lgb.LGBMClassifier(**DEFAULT_PARAMS),
        weight=1,
        verbose=0,
        fit_base_estimator=False,
        target_name="target",
        weight_name="weight",
        target_type=bool,
    ):
        self.clf = clf
        self.weight = weight
        self.verbose = verbose
        self.fit_base_estimator = fit_base_estimator
        self.target_name = target_name
        self.weight_name = weight_name
        self.target_type = target_type

    def fit(self, X, y=None):
        if self.fit_base_estimator:
            self.clf.fit(X, y, verbose=self.verbose)
        return self

    def transform(self, X, y=None):
        self.prob_ = self.clf.predict_proba(X)[:, 1]
        X_good = X.copy(deep=True)
        X_bad = X.copy(deep=True)

        X_good[self.target_name] = self.target_type(False)
        X_good[self.weight_name] = (1 - self.prob_) * self.weight

        X_bad[self.target_name] = self.target_type(True)
        X_bad[self.weight_name] = self.prob_ * self.weight
        result = pd.concat([X_good, X_bad], axis=0)
        return result
