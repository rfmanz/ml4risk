import pandas as pd

################################################
#   align model score/predictions by bad rate
################################################


def get_cum_bad_rate(df, pred_col, target_col):
    df_ = df.copy()

    df_.sort_values(by=[pred_col], ascending=False, inplace=True)
    df_["tot_rank_<TMP>"] = range(len(df_))
    df_["tot_rank_<TMP>"] = df_["tot_rank_<TMP>"] + 1
    df_["bad_rank_<TMP>"] = df_[target_col].cumsum()
    df_["bad_rate"] = df_["bad_rank_<TMP>"] / (df_["tot_rank_<TMP>"])

    assert not df_.bad_rate.isna().any()

    result = df_["bad_rate"]
    return result


def get_score_alignment_table(
    df,
    src_pred,
    tgt_pred,
    target_col,
    src_pred_range=(0, 1),
    tgt_pred_range=(0, 1),
    br_precision=3,
    pred_precision=3,
):
    assert src_pred_range[0] < src_pred_range[1]
    assert tgt_pred_range[0] < tgt_pred_range[1]

    src_br_col = "br_" + src_pred
    tgt_br_col = "br_" + tgt_pred
    src_br_cummax = "br_cummax_" + src_pred
    tgt_br_cummax = "br_cummax_" + tgt_pred

    df = df[[target_col, src_pred, tgt_pred]].copy()

    # clean predictions
    df[tgt_pred] = df[tgt_pred].round(pred_precision)
    df[src_pred] = df[src_pred].round(pred_precision)

    # bad rate for target prediction - 20201130
    br = get_cum_bad_rate(df, tgt_pred, target_col).rename(tgt_br_col)
    df = pd.merge(df, br, left_index=True, right_index=True)

    # bad rate for source prediction - 20191231
    br = get_cum_bad_rate(df, src_pred, target_col).rename(src_br_col)
    df = pd.merge(df, br, left_index=True, right_index=True)

    # get bad rate cummax
    df.sort_values(src_pred, inplace=True, ascending=True)
    df[src_br_cummax] = df[src_br_col].cummax()

    df.sort_values(tgt_pred, inplace=True, ascending=True)
    df[tgt_br_cummax] = df[tgt_br_col].cummax()

    # cumulative bad rate table for the target prediction
    br_tbl_tgt = df[[tgt_pred, tgt_br_cummax]]
    br_tbl_tgt[tgt_br_cummax] = br_tbl_tgt[tgt_br_cummax].round(br_precision)
    br_tbl_tgt = br_tbl_tgt.sort_values(by=tgt_pred).drop_duplicates(
        tgt_br_cummax, keep="first"
    )

    # cumulative bad rate table for the target prediction
    br_tbl_src = df[[src_pred, src_br_cummax]]
    br_tbl_src[src_br_cummax] = br_tbl_src[src_br_cummax].round(br_precision)
    br_tbl_src = br_tbl_src.sort_values(by=src_pred).drop_duplicates(
        src_br_cummax, keep="first"
    )

    # combine cummax table
    br_tbl = pd.merge(
        br_tbl_tgt,
        br_tbl_src,
        left_on=tgt_br_cummax,
        right_on=src_br_cummax,
        how="inner",
    )
    br_tbl["bad_rate"] = br_tbl[tgt_br_cummax]
    br_tbl = br_tbl[["bad_rate", src_pred, tgt_pred]]
    br_tbl.sort_values("bad_rate", inplace=True)

    for col in br_tbl.columns:
        assert br_tbl[col].is_monotonic_increasing

    # drop duplicates and final touch-up
    # to avoid duplicated values
    br_tbl = br_tbl.drop_duplicates(subset=[tgt_pred], keep="first")
    br_tbl = br_tbl.drop_duplicates(subset=[src_pred], keep="first")
    br_tbl = br_tbl.loc[
        br_tbl[tgt_pred] != br_tbl[src_pred], :
    ]  # do we have to keep this?

    first_row = [0.0, src_pred_range[0], tgt_pred_range[0]]
    last_row = [1.0, src_pred_range[1], tgt_pred_range[1]]

    rows = [first_row] + br_tbl.values.tolist() + [last_row]
    br_tbl = pd.DataFrame(rows, columns=br_tbl.columns)

    return br_tbl


def get_aligned_score(df, score_alignment_table, src_col, tgt_col, pred_precision=3):
    """
    df and score_alignment_table must contains a
        src_col and tgt_col as columns

    # NOTE: score_alignment_table must be sorted in ascending order!
    """
    assert score_alignment_table[src_col].is_monotonic_increasing
    assert score_alignment_table[tgt_col].is_monotonic_increasing

    print("source column: ", src_col)
    print("target columns: ", tgt_col)

    tbl = score_alignment_table.copy()

    src_t = tbl[src_col]  # .round(pred_precision)  # source threshold
    src_bins = list(zip(src_t[:-1], src_t[1:]))

    tgt_t = tbl[tgt_col]  # .round(pred_precision)
    tgt_bins = list(zip(tgt_t[:-1], tgt_t[1:]))

    tbl = tbl.iloc[:-1]
    tbl["src_bin"] = src_bins
    tbl["tgt_bin"] = tgt_bins

    cutoffs = src_t
    mapper_dict = {}  # src_label : (src_bin, tgt_bin, bad_rate)
    for idx, row in tbl.iterrows():
        mapper_dict[row["src_bin"]] = (row["src_bin"], row["tgt_bin"], row["bad_rate"])

    score_df = df[src_col].copy().to_frame()
    score_df["bin"] = pd.cut(
        df[src_col],
        cutoffs,
        right=True,
        include_lowest=True,
        labels=tbl["src_bin"].tolist(),
    )

    def map_interpolate(score, src_bin_str):
        (sl, su), (tl, tu), bad_rate = mapper_dict[src_bin_str]
        return (score - sl) / (su - sl) * (tu - tl) + tl

    result = score_df.apply(
        lambda row: map_interpolate(row[src_col], row["bin"]), axis=1
    )

    return result
