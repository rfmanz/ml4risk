import pandas as pd


def validate_payment_str(
    payment_strs: pd.Series,
    origination_date: pd.Series,
    loan_term: pd.Series,
    maturity_date: pd.Series,
) -> bool:
    """Validate the payment_str satisfies our assumption
    of which the string will be updated even if the loan
    has been prepaid/defaulted/deceased...
    """
    raise NotImplemented
