import numpy as np
import pytest
import pandas as pd
import ml4risk.model_design.performance_window as pw
from pandas.testing import assert_series_equal, assert_frame_equal


@pytest.fixture
def loan_df():
    df = pd.DataFrame(
        [
            [1, 0, 0],
            [1, 1, 1],
            [1, 0, 2],
            [1, 1, 3],
            [2, 0, 0],
            [2, 1, 1],
            [2, 1, 2],
            [2, 1, 3],
            [3, 0, 0],
            [3, 0, 1],
            [3, 0, 2],
        ],
        columns=["uid", "target", "seasoning"],
    )
    df["first_bad"] = [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    df["balance"] = [10, 10, 10, 10, 5, 5, 5, 5, 1, 1, 1]
    return df


@pytest.fixture
def agg_df():
    agg = pd.DataFrame()
    agg["seasoning"] = [0, 1, 2, 3]
    agg["nr_bad"] = [0, 2, 0, 0]
    agg["br_inc"] = [0.0, 2 / 3, 0.0, 0.0]
    agg["br_cum"] = [0.0, 2 / 3, 2 / 3, 2 / 3]
    agg["bal_bad"] = [0, 15, 0, 0]
    agg["br_inc_bal"] = [0, 15 / 16, 0, 0]
    agg["br_cum_bal"] = [0, 15 / 16, 15 / 16, 15 / 16]
    return agg


def test_get_first_bad(loan_df):
    first_bad = pw.get_first_bad(loan_df, "uid", "seasoning", "target")
    assert np.array_equal(first_bad, loan_df["first_bad"].values)


def test_get_cum_bad_rate(loan_df, agg_df):
    loan_df["first_bad"] = pw.get_first_bad(loan_df, "uid", "seasoning", "target")
    agg = pw.get_incremental_bads(loan_df, "uid", "seasoning", "first_bad", "balance")
    assert_frame_equal(agg_df, agg)
