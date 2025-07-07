import copy, gc
import numpy as np
import pandas as pd
import lightgbm as lgb
import category_encoders as ce
from sklearn.base import BaseEstimator, TransformerMixin


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


def fit_tree(tree, x, y):
    tree_ = copy.deepcopy(tree)
    pos_tree = tree_.set_params(monotone_constraints=[1])
    pos_tree.fit(x.reshape(-1, 1), y)

    tree_ = copy.deepcopy(tree)
    neg_tree = tree_.set_params(monotone_constraints=[-1])
    neg_tree.fit(x.reshape(-1, 1), y)

    pos_size = len(pos_tree.booster_.trees_to_dataframe())
    neg_size = len(neg_tree.booster_.trees_to_dataframe())

    return pos_tree if pos_size > neg_size else neg_tree


def fit_tree_(x, y, tree, encode_missing="bin"):
    """
    given data, fit the tree with monotonicity enforced
    treat missing either automatically as lightgbm's intrinsic design
    or as its separate bin

    data must be ordinal - this function does not deal with special value
    @params x: 1D np.array
        feature data
    @params y: 1D np.array
        target
    @params missing: str in ["lgb_default", "bin"]
        how to encode missing

    @return tree
    """

    if encode_missing == "lgb_default":
        tree.fit(x.reshape(-1, 1), y)

    elif encode_missing == "bin":
        nan_idx = np.isnan(x)
        x, y = x[~nan_idx], y[~nan_idx]
        tree.fit(x.reshape(-1, 1), y)

    else:
        raise ValueError('encode_missing must be in ["lgb_default", "bin"]')
    return tree


def get_woe_tree_unidir(
    tree, x, y, encode_missing="bin", categorical=[], encode_categorical="bin"
):
    """
    fit tree with methods to deal with categorical values

    @params tree:
        the lightgbm tree used for spliting
    @params x: 1D np.array in floats
        feature data in numerical format. Please encode categorical variables
    @params y: 1D np.array
        target
    @params missing: str in ["lgb_default", "bin"]
        how to encode missing
        if bin: treat missing as its separate bin
        if lgb_default: group missing with any one of the segments
            (lgb's default option to treat missing)
    @params categorical: List
        list of values to treat as categories
    @params encode_categorical: str in ["iterative", "bin"]
        whether to treat categorical as its individual bin or
        iteratively encode it as missing then impute with a numerical value

    @returns tree
        fitted tree
    """
    assert (len(tree.monotone_constraints) == 1) and (
        tree.monotone_constraints[0] in [-1, 1]
    )

    x_ = x.copy().astype(float)
    y_ = y.copy().astype(int)
    tree_ = copy.deepcopy(tree)

    # get location of different indices
    nan_idx = np.isnan(x)
    cat_idx = np.isin(x, categorical)
    num_idx = ~(nan_idx | cat_idx)
    assert nan_idx.sum() + cat_idx.sum() + num_idx.sum() == len(x)

    # case 1: if no categorical and group missing with one of the splited branch
    #    the simplist case
    if (len(categorical) == 0) and (encode_missing == "lgb_default"):
        assert num_idx.sum() + nan_idx.sum() == len(x)
        tree_ = fit_tree_(x_, y_, tree_, encode_missing=encode_missing)

    # case 2: treat missing and categorical as their own bin
    #    arguably the most naive case
    elif (encode_missing == "bin") and (encode_categorical == "bin"):
        tree_ = fit_tree_(
            x_[num_idx], y_[num_idx], tree_, encode_missing=encode_missing
        )

    # case 3: group missing and categorical with splitted branches
    # TODO: should we rethink about imputation and refitting
    elif (encode_categorical == "iterative") and (encode_missing == "lgb_default"):
        raise NotImplemented("Not tested!")
        # setup
        tree_ = copy.deepcopy(tree)
        x_encoded = x_.copy()
        num_idx_ = num_idx and nan_idx
        cat_ = copy.deepcopy(categorical)
        mapping = {}

        # deal with nan
        tree_ = fit_tree_(
            x_encoded[num_idx_], y_[num_idx_], tree_, encode_missing="lgb_default"
        )
        val = impute_nan(tree_, x_encoded, how="mean")
        x_encoded[num_idx_] = val
        mapping[np.nan] = val

        # deal with categoricals
        for c in cat_:
            c_idx = x_encoded == c
            num_idx_ = num_idx_ | c_idx
            x_encoded[c_idx] = np.nan
            tree_ = copy.deepcopy(tree)

            tree_ = fit_tree_(
                x_encoded[num_idx_], y_[num_idx_], tree_, encode_missing="lgb_default"
            )
            val = impute_nan(tree_, x_encoded, how="mean")
            x_encoded[num_idx_] = val
            mapping[c] = val

        # final fitting
        # assert no missing
        tree_ = copy.deepcopy(tree)
        tree_ = fit_tree_(x_encoded, y_, tree_, encode_missing="lgb_default")

    else:
        raise NotImplementedError(
            "such combination of encode_missing and encode_categorical not NotImplemented"
        )

    return tree_


def get_woe_tbl(
    tree,
    x,
    y,
    encode_missing="bin",
    categorical=[],
    encode_categorical="bin",
    min=-np.inf,
    max=np.inf,
):
    """
    fit tree with methods to deal with categorical values

    @params tree:
        the lightgbm tree used for spliting
    @params x: 1D np.array in floats
        feature data in numerical format. Please encode categorical variables
    @params y: 1D np.array
        target
    @params encode_missing: str in ["lgb_default", "bin"]
        how to encode missing
        if bin: treat missing as its separate bin
        if lgb_default: group missing with any one of the segments
            (lgb's default option to treat missing)
    @params categorical: List
        list of values to treat as categories
    @params encode_categorical: str in ["iterative", "bin"]
        whether to treat categorical as its individual bin or
        iteratively encode it as missing then impute with a numerical value
    @params min:
        minimum value
    @params max:
        maximum value

    @returns woe_tbl
        produced woe_tbl
    @returns tree
        fitted tree
    """
    cat = categorical.copy()

    # built positive direction tree
    tree_ = copy.deepcopy(tree)
    tree_.set_params(monotone_constraints=[1])
    pos_tree = get_woe_tree_unidir(
        tree_,
        x,
        y,
        encode_missing=encode_missing,
        categorical=categorical,
        encode_categorical=encode_categorical,
    )

    # build negative direction ree
    tree_ = copy.deepcopy(tree)
    tree_.set_params(monotone_constraints=[-1])
    neg_tree = get_woe_tree_unidir(
        tree_,
        x,
        y,
        encode_missing=encode_missing,
        categorical=categorical,
        encode_categorical=encode_categorical,
    )

    # build woe table for both and return the one that has higher iv
    if encode_missing == "bin":
        cat = sorted(list(set(cat + [np.nan])))
    pos_tbl = build_tbl(
        pos_tree, x, y, encode_missing=encode_missing, categorical=cat, min=min, max=max
    )
    neg_tbl = build_tbl(
        neg_tree, x, y, encode_missing=encode_missing, categorical=cat, min=min, max=max
    )

    if get_iv(pos_tbl) > get_iv(neg_tbl):
        return pos_tbl, pos_tree
    else:
        return neg_tbl, neg_tree


def impute_nan(tree, x, how="mean"):
    """use tree to impute missing in x

    @params tree: lgb.sklearn.LGBMClassifier or lgb.basic.Booster
        tree fitted based on x
    @params x: 1d np.array
        data used for imputation
    @params how: function - currently only supports "mean", "median", "max", "min"
        functions that is used for imputation.
        the function is applied onto the non-missing values that belongs
        to the same bin as the missing values.

    @return value: float
        value used to impute missing
    """
    if isinstance(tree, lgb.sklearn.LGBMClassifier):
        booster = tree.booster_
    elif isinstance(tree, lgb.basic.Booster):
        booster = tree

    else:
        raise ValueError(f"Unknown tree type {type(tree)}")

    # generate prediction and locate the branch of missing inputs
    preds = booster.predict(x.reshape(-1, 1))
    missing_pred = booster.predict(np.array([[np.nan]])).item()
    x_ = x[preds == missing_pred]

    if how in ["mean", "median", "max", "min"]:
        value = getattr(np, f"nan{how}")(x_)
    else:
        raise NotImplementedError("Currently only supporting mean and median")
    return value


def transform(x, woe_tbl):
    """
    given data and woe_tbl, generate woe encoding
    """
    return x


def get_bins(
    x,
    tree,
    encode_missing="lgb_default",
    retbins=False,
    min=-np.inf,
    max=np.inf,
    categorical=[],
):
    """
    get bins given data and tree

    @params x: 1d np.array
        input data
    @params tree: lgb.sklearn.LGBMClassifier or lgb.basic.Booster
        fitted tree

    @return encode: 1d np.array of binned labels
    @return bins: bins
    """
    if isinstance(tree, lgb.sklearn.LGBMClassifier):
        lgbm_booster = tree.booster_
    elif isinstance(tree, lgb.basic.Booster):
        lgbm_booster = tree
    else:
        raise ValueError(f"Unknown tree type {type(tree)}")

    # produce bins
    bins = sorted(lgbm_booster.trees_to_dataframe().threshold.unique())
    bins.extend([min, max])
    bins = sorted([i for i in bins if not np.isnan(i)])

    encoded = np.zeros_like(x) - 1
    if encode_missing == "lgb_default":
        encoded = np.digitize(x, bins, right=True)

        # can actually use impute missing to do this
        preds = lgbm_booster.predict(x.reshape(-1, 1))
        missing_pred = lgbm_booster.predict(np.array([[np.nan]])).item()
        encoded_groups = np.unique(encoded[preds == missing_pred])
        encoded[np.isin(encoded, encoded_groups)] = encoded_groups.min()

    elif encode_missing == "bin":
        nan_idx = np.isnan(x)
        encoded[~nan_idx] = np.digitize(x[~nan_idx], bins, right=True)
        bins.append(np.nan)

    else:
        raise NotImplementedError("encode_missing but be in ['lgb_default', 'bin']")

    if retbins:
        return encoded.astype(int), bins
    return encoded.astype(int)


#######################################
#       build table given tree
#######################################


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
    num_idx = (~nan_idx) & (~cat_idx) & (x <= max) & (x >= min)

    # numbers need to align
    assert num_idx.sum() + cat_idx.sum() + nan_idx.sum() == len(x)

    # process numerical and categorical values separately
    if encode_missing == "bin":
        num_tbl = build_num_tbl(booster, x[num_idx], y[num_idx], min=min, max=max)
        cat_tbl = build_cat_tbl(x[cat_idx | nan_idx], y[cat_idx | nan_idx])

    elif encode_missing == "lgb_default":
        num_tbl = build_num_tbl(
            booster, x[num_idx | nan_idx], y[num_idx | nan_idx], min=min, max=max
        )
        cat_tbl = build_cat_tbl(x[cat_idx], y[cat_idx])
    else:
        raise ValueError("encode_missing must in ['bin', 'lgb_default']")

    # combine numerical and categorical tables
    cols = ["min", "max", "type", "bin", "label", "event", "nonevent"]
    tbl = pd.concat([num_tbl[cols], cat_tbl[cols]], axis=0).reset_index(drop=True)

    tbl["counts"] = tbl["event"] + tbl["nonevent"]
    nr_event_nonevent = tbl[["label", "event", "nonevent"]].drop_duplicates()
    tbl["dist_event"] = tbl["event"].astype(float) / nr_event_nonevent["event"].sum()
    tbl["dist_nonevent"] = (
        tbl["nonevent"].astype(float) / nr_event_nonevent["nonevent"].sum()
    )

    tbl["woe"] = np.log(tbl["dist_event"] / tbl["dist_nonevent"])
    tbl["iv"] = (tbl["dist_event"] - tbl["dist_nonevent"]) * tbl["woe"]

    return tbl


def get_missing_bin(booster, x, y, threshold):
    """find the"""

    encoded = np.digitize(x, threshold, right=True)
    preds = booster.predict(x.reshape(-1, 1))
    missing_pred = booster.predict(np.array([[np.nan]])).item()
    encoded_groups = np.unique(encoded[preds == missing_pred])

    return encoded_groups.min()


def build_num_tbl(booster, x, y, min=-np.inf, max=np.inf):
    if len(x) == 0:
        cols = ["min", "max", "type", "bin", "label", "event", "nonevent"]
        return pd.DataFrame(columns=cols)

    threshold = sorted(booster.trees_to_dataframe().threshold.astype(float).unique())
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
    woe_df = pd.DataFrame(np.nan, index=np.arange(len(x)), columns=["bin"])

    woe_df[~np.isnan(x)] = pd.DataFrame(
        pd.cut(
            x[~np.isnan(x)],
            bins=threshold,
            right=True,
            include_lowest=True,
            labels=tbl["bin"],
        )
    ).values

    if np.isnan(x).any():
        bin_missing = get_missing_bin(booster, x, y, threshold)
        tbl = tbl.append(
            {
                "min": np.nan,
                "max": np.nan,
                "type": "numerical",
                "bin": bin_missing,
                "label": "num_bin_" + str(bin_missing),
            },
            ignore_index=True,
        )

        woe_df[np.isnan(x)] = bin_missing

    woe_df["y"] = y

    # event and nonevent counts are duplicated here for NaN
    # because we used left join
    woe_tbl = woe_df[["bin", "y"]].value_counts(dropna=False).unstack()
    woe_tbl.rename({1.0: "event", 0.0: "nonevent"}, axis=1, inplace=True)
    tbl = pd.merge(tbl, woe_tbl, on="bin", how="left")
    tbl[["event", "nonevent"]] = tbl[["event", "nonevent"]].fillna(0)

    return tbl


def build_cat_tbl(x, y):
    if len(x) == 0:
        cols = ["min", "max", "type", "bin", "label", "event", "nonevent"]
        return pd.DataFrame(columns=cols)

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

    woe_tbl = woe_df[["bin", "y"]].value_counts(dropna=False).unstack()
    woe_tbl.rename({1.0: "event", 0.0: "nonevent"}, axis=1, inplace=True)

    tbl = pd.merge(tbl, woe_tbl, on="bin", how="left")
    tbl[["event", "nonevent"]] = tbl[["event", "nonevent"]].fillna(0)

    return tbl


def get_iv(woe_tbl):
    """
    "iv", "label", must be in woe_df.columns
    """
    return woe_tbl[["iv", "label"]].drop_duplicates()["iv"].sum()
