import copy, gc
import numpy as np
import pandas as pd
import lightgbm as lgb
import category_encoders as ce
from sklearn.base import BaseEstimator, TransformerMixin


#################################
#          Build Tree
#################################


def get_default_tree(min_bin_size, direction):
    """
    get default tree for splitting

    @params min_bin_size: int
        minimum number of data points to have in each bin
    @params direction: int in [-1, 0, 1]
        1  : enforce upward trend
        -1 : enforce downward trend
        0  : no enforcement

    @returns tree
    """
    tree = lgb.LGBMClassifier(
        num_leaves=100,
        max_depth=-1,
        n_estimators=1,
        learning_rate=0.01,
        random_state=42,
        min_child_samples=min_bin_size,
        monotone_constraints=[direction],
    )
    return tree


#################################
#         Build Table
#################################


def build_tbl(
    tree, x, y, encode_missing="bin", categorical=[], min=-np.inf, max=np.inf
):
    """
    given data and tree tree, generate woe table
    """
    tbl = pd.DataFrame()

    if isinstance(tree, lgb.sklearn.LGBMClassifier):
        booster = tree.booster_
    elif isinstance(tree, lgb.basic.Booster):
        booster = tree
    else:
        raise ValueError(f"Unknown tree type {type(tree)}")

    nan_idx = np.isnan(x)
    cat_idx = np.isin(x, categorical)
    if encode_missing == "bin":
        cat_idx = cat_idx | nan_idx

    num_idx = (~cat_idx) & (x <= max) & (x >= min)

    # numbers need to align
    assert (num_idx.sum() + cat_idx.sum() == len(x)) or (
        num_idx.sum() + cat_idx.sum() + nan_idx.sum() == len(x)
    )

    cols = ["min", "max", "type", "bin", "label", "woe", "counts"]
    num_tbl = build_num_tbl(booster, x[num_idx], y[num_idx])
    cat_tbl = build_cat_tbl(x[cat_idx], y[cat_idx])

    tbl = pd.concat([num_tbl[cols], cat_tbl[cols]], axis=0).reset_index(drop=True)
    return tbl


def build_num_tbl(booster, x, y, min=-np.inf, max=np.inf):
    threshold = sorted(booster.trees_to_dataframe().threshold.unique())
    threshold.extend([min, max])
    threshold = sorted([i for i in threshold if not np.isnan(i)])
    min_col, max_col = threshold[:-1], threshold[1:]

    tbl = pd.DataFrame()
    tbl["min"] = min_col
    tbl["max"] = max_col
    tbl["type"] = "numerical"
    tbl["bin"] = tbl.index
    tbl["label"] = "num_bin_" + tbl.index.astype(str)

    # build WOE portion
    woe_df = pd.DataFrame(
        pd.cut(x, bins=threshold, right=True, include_lowest=True, labels=tbl["bin"])
    )
    woe_df.columns = ["bin"]
    woe_df["y"] = y

    woe_encoder = ce.woe.WOEEncoder()
    woe_df["woe"] = woe_encoder.fit_transform(woe_df["bin"], woe_df["y"])
    woe_tbl = woe_df[["bin", "woe"]].value_counts().rename("counts").reset_index()
    tbl = pd.merge(tbl, woe_tbl, on="bin", how="left")

    return tbl


def build_cat_tbl(x, y):

    tbl = pd.Series(x).value_counts(dropna=False).to_frame()
    tbl["min"] = tbl.index.values
    tbl["max"] = tbl.index.values
    tbl["type"] = "categorical"
    tbl["bin"] = tbl.index
    tbl["label"] = "cat_bin_" + tbl.index.astype(str)

    # build WOE
    woe_df = pd.DataFrame(x)
    woe_df.columns = ["bin"]
    woe_df["x_encoded"] = woe_df["bin"].astype(str)
    woe_df["y"] = y

    woe_encoder = ce.woe.WOEEncoder()
    woe_df["woe"] = woe_encoder.fit_transform(woe_df["x_encoded"], woe_df["y"])
    woe_tbl = (
        woe_df[["bin", "woe"]].value_counts(dropna=False).rename("counts").reset_index()
    )
    tbl = pd.merge(tbl, woe_tbl, on="bin", how="left")

    return tbl
