import numpy as np
import pandas as pd
import lightgbm as lgb
import ml4risk.model_development.woe_lgbm as wl
from sklearn.base import BaseEstimator, TransformerMixin


min_bin_size = 15000
DEFAULT_TREE = wl.get_default_tree(15000, -1)


class LgbWOEEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, tree=DEFAULT_TREE):
        self.tree = tree

    def fit(
        self,
        X,
        y=None,
        encode_missing="bin",
        categorical=[],
        encode_categorical="bin",
        min=-np.inf,
        max=np.inf,
    ):
        if isinstance(X, pd.Series):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        tree = wl.get_default_tree(15000, -1)
        self.woe_tbl, self.tree = wl.get_woe_tbl(
            tree,
            X,
            y,
            encode_missing=encode_missing,
            categorical=categorical,
            min=min,
            max=max,
        )

        return self
