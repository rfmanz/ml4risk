import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging, sys


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    fmt="%(asctime)s: %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p"
)
if not logger.hasHandlers():
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)


def get_first_bad(
    df: pd.DataFrame,
    id_col: str,
    seasoning_col: str,
    bad_col: str,
) -> np.array:
    """get indicator first seasoning an id turned bad

    Parameters
    ----------
    df: pd.DataFrame
        monthly loan records that contains id_col, seasoning_col, first_bad_col
    id_col: str
        column contains loan id
    seasoning_col: str
        column contains loan seasoning/month-on-book
    bad_col: str
        column indicating the loan status. needs to be binary / bool

    Returns
    -------
    df: pd.DataFrame

    """

    df = df[[id_col, seasoning_col, bad_col]].copy()
    first_bad = (
        df[df[bad_col].astype(bool)].groupby(id_col)[seasoning_col].min().reset_index()
    )
    first_bad["first_bad_"] = 1
    df = pd.merge(df, first_bad, on=[id_col, seasoning_col], how="left")
    result = df["first_bad_"].fillna(0).astype(int)
    return result.values


def get_incremental_bads_(
    df: pd.DataFrame,
    id_col: str,
    seasoning_col: str,
    first_bad_col: str,
    bal_col: str = None,
) -> pd.DataFrame:
    """get incremental bad counts

    if bal_col provided, calculate incremental bad in terms of balance as well

    Parameters
    ----------
    df: pd.DataFrame
        monthly loan records that contains id_col, seasoning_col, first_bad_col
    id_col: str
        column contains loan id
    seasoning_col: str
        column contains loan seasoning/month-on-book
    first_bad_col: str
        column contains the indicator suggesting the loan turned bad on this month
        compute using function `get_first_bad`

    Returns
    -------
    df: pd.DataFrame

    """

    # setup
    assert set(df[first_bad_col].unique()).issubset({0, 1})
    cols = [id_col, seasoning_col, first_bad_col]
    if bal_col:
        cols.append(bal_col)
    df = df[cols].copy()

    # get total counts/balances
    nr_loan = df[id_col].nunique()

    # get incremental values
    agg = pd.DataFrame()
    agg["nr_bad"] = df.groupby(seasoning_col)[first_bad_col].sum()
    agg.sort_index(inplace=True)  # sort by seasoning
    agg["br_inc"] = agg["nr_bad"] / nr_loan
    agg["br_cum"] = agg["br_inc"].cumsum()

    # if bal_col provided, calculate bad in $
    if bal_col:
        df_init_term = (
            df.sort_values(by=seasoning_col).groupby(id_col).first()
        )  # first record of every account
        total_bal = df_init_term[bal_col].sum()
        df["bal_first_bad_"] = df[first_bad_col] * df[bal_col]

        agg["bal_bad"] = df.groupby(seasoning_col)["bal_first_bad_"].sum()
        agg["br_inc_bal"] = agg["bal_bad"] / total_bal
        agg["br_cum_bal"] = agg["br_inc_bal"].cumsum()

    return agg.reset_index()


def get_incremental_bads(
    df: pd.DataFrame,
    id_col: str,
    seasoning_col: str,
    first_bad_col: str,
    bal_col: str = None,
    group_col: str = None,
    segments: dict = None,
) -> pd.DataFrame:
    """get_incremental_bads, with group

    Parameters
    ----------
    df: pd.DataFrame
        monthly loan records that contains id_col, seasoning_col, first_bad_col
    id_col: str
        column contains loan id
    seasoning_col: str
        column contains loan seasoning/month-on-book
    first_bad_col: str
        column contains the indicator suggesting the loan turned bad on this month
        compute using function `get_first_bad`
    group_col: str
        column to group by

    Returns
    -------
    df: pd.DataFrame

    """

    if not group_col and not segments:
        return get_incremental_bads_(df, id_col, seasoning_col, first_bad_col, bal_col)

    elif group_col:
        dfs = []
        for v in tqdm(df[group_col].unique()):
            df_ = df[df[group_col] == v]
            df_ = get_incremental_bads(
                df_, id_col, seasoning_col, first_bad_col, bal_col
            )
            df_[group_col] = v
            dfs.append(df_)
        return pd.concat(dfs, axis=0).reset_index(drop=True)

    elif segment:
        raise NotImplemented

    raise NotImplemented


def plot_cum_curve(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_col: str = None,
    fig=None,
    ax=None,
    figsize=None,
    **kwargs,
):
    """plot cumm curve: bad rate vs. seasoning

    Parameters
    ----------
    df: pd.DataFrame
        dataframe containing columns x_col, y_col, and maybe groups
    x_col: str
        column name indicating values to plot on the x axis
        e.g. seasoning
    y_col: str
        column name indicating values to plot on y axis
        e.g. cumulative bad rate
    groups: str
        column name, which will be grouped with each group plotted individually
    fig:
        matplotlib figure
    ax: matplotlib.axes.Axes
        pre-existing axes for the plot
    **kwargs: dict
        parameters for matplotlib.pyplot.plot()

    Returns
    -------
    fig:
        matplotlib figure
    ax:
        matplotlib axis
    """
    from tqdm import tqdm

    assert (x_col in df.columns) and (y_col in df.columns)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if group_col:
        for l in tqdm(sorted(df[group_col].unique())):
            df_ = df[df[group_col] == l]
            if "label" in kwargs:
                logger.warning(
                    "Overwritting provided label since we have multiple groups"
                )
            kwargs["label"] = f"{group_col} = {l}"
            ax.plot(df_[x_col], df_[y_col], **kwargs)

    else:
        ax.plot(df[x_col], df[y_col], **kwargs)

    return fig, ax
