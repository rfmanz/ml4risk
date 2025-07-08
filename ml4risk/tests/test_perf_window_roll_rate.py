import pytest
import pandas as pd
import ml4risk.model_design.performance_window as pw


def test_get_loan_status():
    assert pw.get_loan_status("1") == "1-29DPD"
    assert pw.get_loan_status("2") == "1-29DPD"
    assert pw.get_loan_status("3") == "30-59DPD"
    assert pw.get_loan_status("4") == "60-89DPD"
    assert pw.get_loan_status("5") == "90-119DPD"
    for s in ["6", "B", "D", "W"]:
        assert pw.get_loan_status(s) == "DEFAULT"

    for s in ["", "~", "Z", " ", "c"]:
        assert pw.get_loan_status(s) == "_UNK_"


@pytest.fixture
def df():
    pstrs = ["CC", "CCCD", "CDDDDD", "CCCCDDDD"]
    # k = 1
    result1 = ["CURRENT", "CURRENT", "CURRENT", "CURRENT"]
    # k = 3
    result2 = ["_MISSING_", "CURRENT", "DEFAULT", "CURRENT"]
    # k = 0
    # k = 100
    result3 = ["_MISSING_", "_MISSING_", "_MISSING_", "_MISSING_"]

    df = pd.DataFrame(
        {
            "payment_str": pstrs,
            "result1": result1,
            "result2": result2,
            "result3": result3,
        }
    )
    return df


@pytest.fixture
def big_df():
    pstrs = ["CC", "CCCD", "C123", "223DDDDD", "C6D", "2WWW", "CBBB", "CDDD"]
    return pd.DataFrame({"status_string": pstrs})


def test_payment_status_money_k(df):
    from pandas.testing import assert_series_equal

    assert_series_equal(
        pw.get_payment_status_month_k(df.payment_str, 1), df.result1, check_names=False
    )
    assert_series_equal(
        pw.get_payment_status_month_k(df.payment_str, 3), df.result2, check_names=False
    )
    assert_series_equal(
        pw.get_payment_status_month_k(df.payment_str, 100),
        df.result3,
        check_names=False,
    )

    with pytest.raises(AssertionError):
        pw.get_payment_status_month_k(df.payment_str, 0)


def test_status_k_after_n(df):
    from pandas.testing import assert_series_equal

    s1, s2 = pw.get_status_k_after_n(df.payment_str, 2, 1)
    assert_series_equal(s1, df.result1, check_names=False)
    assert_series_equal(s2, df.result2, check_names=False)

    s1, s2 = pw.get_status_k_after_n(df.payment_str, 97, 3)
    assert_series_equal(s1, df.result2, check_names=False)
    assert_series_equal(s2, df.result3, check_names=False)

    s1, s2 = pw.get_status_k_after_n(df.payment_str, 99, 1)
    assert_series_equal(s1, df.result1, check_names=False)
    assert_series_equal(s2, df.result3, check_names=False)


def test_status_transition_table(big_df):
    from pandas.testing import assert_frame_equal

    big_df["status1"] = pw.get_payment_status_month_k(big_df["status_string"], 1)
    big_df["status2after1"] = pw.get_payment_status_month_k(big_df["status_string"], 3)

    tbl1 = pw.get_status_transition_table(
        big_df, "status1", "status2after1", margins=True
    )
    result1 = pd.DataFrame(
        {
            "1-29DPD": {"1-29DPD": 0, "CURRENT": 1, "All": 1},
            "30-59DPD": {"1-29DPD": 1, "CURRENT": 0, "All": 1},
            "CURRENT": {"1-29DPD": 0, "CURRENT": 1, "All": 1},
            "DEFAULT": {"1-29DPD": 1, "CURRENT": 2, "All": 3},
            "_MISSING_": {"1-29DPD": 0, "CURRENT": 2, "All": 2},
            "All": {"1-29DPD": 2, "CURRENT": 6, "All": 8},
        }
    )

    assert_frame_equal(tbl1, result1, check_names=False)

    tbl2 = pw.get_status_transition_table(
        big_df, "status1", "status2after1", margins=False
    )
    result2 = pd.DataFrame(
        {
            "1-29DPD": {"1-29DPD": 0, "CURRENT": 1},
            "30-59DPD": {"1-29DPD": 1, "CURRENT": 0},
            "CURRENT": {"1-29DPD": 0, "CURRENT": 1},
            "DEFAULT": {"1-29DPD": 1, "CURRENT": 2},
            "_MISSING_": {"1-29DPD": 0, "CURRENT": 2},
        }
    )
    assert_frame_equal(tbl2, result2, check_names=False)

    tbl3 = pw.get_status_transition_table(
        big_df, "status1", "status2after1", margins=True, normalize=True
    )
    result3 = pd.DataFrame(
        {
            "1-29DPD": {"1-29DPD": 0, "CURRENT": 1 / 6, "All": 1 / 8},
            "30-59DPD": {"1-29DPD": 1 / 2, "CURRENT": 0, "All": 1 / 8},
            "CURRENT": {"1-29DPD": 0, "CURRENT": 1 / 6, "All": 1 / 8},
            "DEFAULT": {"1-29DPD": 1 / 2, "CURRENT": 2 / 6, "All": 3 / 8},
            "_MISSING_": {"1-29DPD": 0, "CURRENT": 2 / 6, "All": 2 / 8},
            "All": {"1-29DPD": 1, "CURRENT": 1, "All": 1},
        }
    ).astype(float)

    assert_frame_equal(tbl3, result3, check_names=False)
