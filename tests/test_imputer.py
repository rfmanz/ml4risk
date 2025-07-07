import numpy as np
import pytest
import pandas as pd
import ml4risk.data_preparation.imputer as ip


@pytest.fixture
def pos_woe_tbl():
    # mc inc
    data = [
        ["20.0%", 1, 1.9, 0.1],
        ["20.0%", 2, 2.9, 0.2],
        ["20.0%", 3, 3.9, 0.3],
        ["20.0%", 4, 4.9, 0.4],
        ["20.0%", np.nan, np.nan, 0.28],
    ]
    index = [0, 1, 2, 3, "missing"]
    columns = ["%accts", "min", "max", "woe"]
    pos_eg = pd.DataFrame(data=data, index=index, columns=columns)
    return pos_eg


@pytest.fixture
def neg_woe_tbl():
    # mc dec
    data = [
        ["20.0%", 1, 1.9, 0.4],
        ["20.0%", 2, 2.9, 0.3],
        ["20.0%", 3, 3.9, 0.2],
        ["20.0%", 4, 4.9, 0.1],
        ["20.0%", np.nan, np.nan, 0.28],
    ]
    index = [0, 1, 2, 3, "missing"]
    columns = ["%accts", "min", "max", "woe"]
    neg_eg = pd.DataFrame(data=data, index=index, columns=columns)
    return neg_eg


@pytest.mark.parametrize("missing_woe,imputed_val", [(0.28, 3), (0.23, 2.9)])
def test_get_impute_val_closest_pos(pos_woe_tbl, missing_woe, imputed_val):
    pos_woe_tbl.loc["missing", "woe"] = missing_woe
    assert ip.get_missing_impute_value(pos_woe_tbl, "closest_boundary") == imputed_val


@pytest.mark.parametrize("missing_woe,imputed_val", [(0.28, 2.9), (0.23, 3)])
def test_get_impute_val_closest_neg(neg_woe_tbl, missing_woe, imputed_val):
    neg_woe_tbl.loc["missing", "woe"] = missing_woe
    assert ip.get_missing_impute_value(neg_woe_tbl, "closest_boundary") == imputed_val


@pytest.mark.parametrize("missing_woe,imputed_val", [(0.28, 3.45), (0.23, 2.45)])
def test_get_impute_val_midpoint_pos(pos_woe_tbl, missing_woe, imputed_val):
    pos_woe_tbl.loc["missing", "woe"] = missing_woe
    assert ip.get_missing_impute_value(pos_woe_tbl, "midpoint") == imputed_val


@pytest.mark.parametrize("missing_woe,imputed_val", [(0.28, 2.45), (0.23, 3.45)])
def test_get_impute_val_midpoint_neg(neg_woe_tbl, missing_woe, imputed_val):
    neg_woe_tbl.loc["missing", "woe"] = missing_woe
    assert ip.get_missing_impute_value(neg_woe_tbl, "midpoint") == imputed_val
