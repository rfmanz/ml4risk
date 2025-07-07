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


def build_tbl(tree, x, encode_missing="bin", categorical=[], min=-np.inf, max=np.inf):
    """
    given data and tree tree, generate woe table
    """
    tbl = pd.DataFrame()

    if isinstance(tree, lgb.sklearn.LGBMClassifier):
        lgbm_booster = tree.booster_
    elif isinstance(tree, lgb.basic.Booster):
        lgbm_booster = tree
    else:
        raise ValueError(f"Unknown tree type {type(tree)}")

    # build numerical part of the df
    # lightgbm process thresholds with '<='
    threshold = sorted(lgbm_booster.trees_to_dataframe().threshold.unique())
    threshold.extend([min, max])
    threshold = sorted([i for i in threshold if not np.isnan(i)])
    min_col, max_col = threshold[:-1], threshold[1:]
    types = ["numerical" for i in range(len(min_col))]

    # categorical
    if encode_missing == "bin":
        min_col.append(np.nan)
        max_col.append(np.nan)
        types.append("categorical")

    min_col.extend(categorical)
    max_col.extend(categorical)
    types.extend(["categorical" for i in range(len(categorical))])

    tbl["min"] = min_col
    tbl["max"] = max_col
    tbl["type"] = types

    return tbl
