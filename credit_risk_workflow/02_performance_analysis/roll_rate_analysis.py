import pandas as pd
from collections import defaultdict as ddict
from .validation import validate_payment_str


##########################################
#             default params
##########################################


def get_loan_status(status_code: str) -> str:
    """get loan status from the perf_var/encoded str

    @params status_code - str
        individual char indicating status for payment strings
    @returns status - str
        status grouped for business strategy purposes
    """
    loan_status = {
        "1": "1-9 DAYS DELNQ",
        "2": "10-29 DAYS DELNQ",
        "3": "30-59 DAYS DELNQ",
        "4": "60-89 DAYS DELNQ",
        "5": "90-119 DAYS DELNQ",
        "6": "120+ DAYS DELNQ",
        "B": "BANKRUPTCY",
        "C": "IN REPAY",
        "D": "CHARGE OFF PIF",
        "E": "ERROR",
        "F": "FORBEARANCE",
        "G": "REFI GRACE",
        "H": "HARDSHIP/OTHER DEFERMENT",
        "N": "DEATH ALLEGED",
        "M": "DEATH VERIFIED",
        "P": "PAID IN FULL - NEGATIVE BALANCE",
        "S": "IN SCHOOL",
        "U": "UNBOARDED",
        "W": "WRITE OFF",
        "X": "PENDING FORBEARANCE",
        "Y": "NOT FULLY ORIGINATED",
    }

    mapper = {
        "C": "CURRENT",
        "1": "1-29DPD",
        "2": "1-29DPD",
        "3": "30-59DPD",
        "4": "60-89DPD",
        "5": "90-119DPD",
        "6": "DEFAULT",
        "B": "DEFAULT",
        "D": "DEFAULT",
        "W": "DEFAULT",
    }

    mapper = {**dict([(k, k) for k in loan_status.keys()]), **mapper}
    mapper = ddict(lambda: "_UNK_", mapper)

    return mapper[status_code]


def get_payment_status_month_k(
    status_strings: pd.Series,
    k: int,
    return_last=False,
    status_fn=get_loan_status,
) -> pd.Series:
    """low level function to get loan status from payment string at month k

    @params status_strings - pd.Series
        sequential payment str in Series.
        notice, at the current point of development, the payment_str
        keeps being updated even after prepayment or default.
    @params k - int
        get kth month's loan status, 1 <= k < len(payment_str)
    @params return_last - bool
        whether to return last status if len(status_string) < k.
        retuan_last = True returns "_MISSING_" as default
        return_last = True returns the last char of the status string
    @params status_fn - function(int) -> int
        customized mapper function that maps loan's status code to another value
        default = get_loan_status
    """
    assert k > 0

    def helper(pstr):
        if k >= len(pstr) and len(pstr) > 0:
            if return_last:
                return status_fn(pstr[-1])
            return "_MISSING_"
        return status_fn(pstr[k - 1])

    result = status_strings.apply(helper)
    return result


def get_status_k_after_n(status_strings: pd.Series, k: int, n: int):
    """get loan status k months after n monthly payments"""
    status_n = get_payment_status_month_k(status_strings, n)
    status_nplusk = get_payment_status_month_k(status_strings, n + k)
    return status_n, status_nplusk


def get_status_transition_table(
    df, col1, col2, exclusion=[], normalize=False, margins=True
) -> pd.DataFrame:
    """get status transition table

    @params df - pd.DataFrame
        df that must contain col1 and col2
    @params col1 - str
        source column, as index for the transtion table
    @params col2 - str
        destination column, as column for the transition table
    @params exclusion - list
        list of status code to exclude from showing
    @params normalize - bool
        whether to provide normalized percentages
    @params margins - bool
        whether to include margins
    """

    df = df[(~df[col1].isin(exclusion)) & (~df[col2].isin(exclusion))]
    counts_tbl = pd.crosstab(index=df[col1], columns=df[col2], margins=margins)

    if not normalize:
        return counts_tbl

    pct_tbl = counts_tbl.div(counts_tbl.iloc[:, -1], axis=0)
    return pct_tbl
