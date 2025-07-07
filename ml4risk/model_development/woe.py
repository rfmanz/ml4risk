# this is module will be used to calcuate woe and do data woe transformation
# auther: bo shao (bshao@sofi.org)
# copyright: Social Finance
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
from colorama import Fore, Back, Style
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import scipy.stats.stats as stats
import sys
from math import sqrt
import json


### weighted pearson correlation
def wpearson(vec_1, vec_2, weights, r=4):
    list_length = len(vec_1)
    try:
        weights = list(map(float, weights))
    except:
        print("Invalid weights.")
        sys.exit(1)

    try:
        vec_1 = list(map(float, vec_1))
        vec_2 = list(map(float, vec_2))
        if any(len(x) != list_length for x in [vec_2, weights]):
            print("Vector/Weight sizes not equal.")
            sys.exit(1)
    except:
        print("Invalid vectors.")
        sys.exit(1)

    # Find total weight sum.
    w_sum = sum(weights)
    # Calculate the weighted average relative value of vector 1 and vector 2.
    vec1_sum = 0.0
    vec2_sum = 0.0
    for x in range(len(vec_1)):
        vec1_sum += weights[x] * vec_1[x]
        vec2_sum += weights[x] * vec_2[x]
    vec1_avg = vec1_sum / w_sum
    vec2_avg = vec2_sum / w_sum

    # Calculate wPCC
    sum_top = 0.0
    sum_bottom1 = 0.0
    sum_bottom2 = 0.0

    for x in range(len(vec_1)):
        dif_1 = vec_1[x] - vec1_avg
        dif_2 = vec_2[x] - vec2_avg
        sum_top += weights[x] * dif_1 * dif_2
        sum_bottom1 += (dif_1 ** 2) * (weights[x])
        sum_bottom2 += (dif_2 ** 2) * (weights[x])
    if (sum_bottom1 * sum_bottom2) == 0:
        cor = 2
    else:
        cor = sum_top / (sqrt(sum_bottom1 * sum_bottom2))
    return round(cor, r)


###  helper to get direction of monotonicity given woe_dict in WOE_Transform class
def get_monotone_dir(woe_dict):
    result = {}
    for k in woe_dict:
        tbl = woe_dict[k]
        if tbl.iloc[0]["woe"] < tbl.iloc[1]["woe"]:
            direction = 1
        else:
            direction = -1

        result[k] = direction
    return result


### WOE Caluation and Transformation Class
class WOE_Transform(object):

    # initiate values
    def __init__(
        self, method="tree", num_bin_start=40, min_samples_leaf=500, min_iv=0.01
    ):
        self.method = method
        self.num_bins = num_bin_start
        self.min_iv = min_iv
        self.min_samples_leaf = min_samples_leaf
        self.num_list_local = []
        self.char_list_local = []
        self.num_list = []
        self.char_list = []
        self.invalid_list = []
        self.cutoffs = {}
        self.iv = {}
        self.iv_detail = {}
        self.woe = {}
        self.special = {}
        self.psi = {}
        self.psi_detail = {}
        self.psi_check = {}

    # check the datatype of Y and X
    def _check(self, X, Y):

        if isinstance(X, pd.DataFrame) == False:
            raise ValueError("X is not a dataframe")

        dtype = X.dtypes
        num_list = dtype[(dtype == "int64") | (dtype == "float64")].index.to_list()
        char_list = dtype[
            (dtype == "object") | (dtype == "bool") | (dtype == "category")
        ].index.to_list()
        time_list = dtype[
            (dtype == "datetime64[ns]") | (dtype == "datetime64")
        ].index.to_list()

        if time_list:
            print(
                Fore.GREEN + "datetime attributes are removed from this step:",
                Style.RESET_ALL,
            )
            print(time_list)
            print()

        # check attr with high missing
        df_missing = X[num_list].isnull().sum() / len(X) > 0.99
        num_missing_list = df_missing[df_missing == True].index.to_list()
        if num_missing_list:
            print(
                Fore.GREEN + "Attrs removed--missing pct>99%: ",
                Style.RESET_ALL,
                end=" ",
            )
            print(num_missing_list)

        for attr in num_missing_list:
            num_list.remove(attr)

        self.num_list_local = num_list
        self.char_list_local = char_list

        for num in num_list:
            if num not in self.num_list:
                self.num_list.append(num)

        for char in char_list:
            if char not in self.char_list:
                self.char_list.append(char)

        if Y.dtypes not in ["int64", "float64"]:
            raise ValueError("Target variable is non-numerical")
        elif Y.isnull().sum() > 0:
            raise ValueError("missing values are found in target variable")

        else:
            return self

    def _check_transform(self, X):

        dtype = X.dtypes
        num_list = dtype[(dtype == "int64") | (dtype == "float64")].index.to_list()
        char_list = dtype[
            (dtype == "object") | (dtype == "bool") | (dtype == "category")
        ].index.to_list()

        if (
            set(self.num_list).issubset(set(num_list)) == True
            and set(self.char_list).issubset(set(char_list)) == True
        ):
            a = 1

        elif (
            set(num_list).issubset(set(self.num_list)) == True
            and set(char_list).issubset(set(self.char_list)) == True
        ):
            a = 1

        else:
            raise ValueError("Attributes do not match")

    # woe for Char
    def _bin_char(self, X, Y, Y_weight):
        i = 0
        XX = X.copy()
        to_remove = []

        for char in self.char_list_local:
            i = i + 1

            if XX[char].nunique() > 100:
                to_remove.append(char)
                print(
                    Fore.RED + char,
                    Style.RESET_ALL,
                    " has more than 100 categories, will be ignored",
                )
            else:
                XX[char].astype("object")
                # if a catetory with <=500, assign to "other"
                lessmin = XX[char].value_counts() < self.min_samples_leaf
                lessmin_list = lessmin[lessmin == True].index.to_list()
                XX[char + "_t"] = XX[char].apply(
                    lambda x: "0:Other" if x in lessmin_list else x
                )
                XX.loc[XX[char + "_t"].isnull(), char + "_t"] = "1:Missing"

                df_combine = pd.concat([Y, Y_weight, XX[char + "_t"]], axis=1)
                df_combine.columns = ["target", "weight", "x"]

                # cacluate statistics
                # num_accts=df_combine['x'].value_counts()
                num_accts = df_combine.groupby("x")["weight"].sum()
                pct_accts = num_accts / df_combine["weight"].sum()

                num_good = round(
                    df_combine[df_combine["target"] == 0].groupby("x")["weight"].sum(),
                    1,
                )
                num_bad = round(
                    df_combine[df_combine["target"] == 1].groupby("x")["weight"].sum(),
                    1,
                )

                dist_good = (
                    num_good / df_combine[df_combine["target"] == 0]["weight"].sum()
                )
                dist_bad = (
                    num_bad / df_combine[df_combine["target"] == 1]["weight"].sum()
                )
                target_rates = num_bad / num_accts
                woe = np.log(dist_bad / dist_good)
                iv = (dist_bad - dist_good) * woe

                final_bin = pd.concat(
                    [
                        num_accts,
                        pct_accts,
                        num_good,
                        num_bad,
                        dist_good,
                        dist_bad,
                        target_rates,
                        woe,
                        iv,
                    ],
                    axis=1,
                    sort=True,
                )

                final_bin.columns = [
                    "#accts",
                    "%accts",
                    "#good",
                    "#bad",
                    "dist_good",
                    "dist_bad",
                    "target_rate",
                    "woe",
                    "iv",
                ]

                final_bin["attr"] = char
                final_bin["%accts"] = final_bin["%accts"].map("{:,.2%}".format)
                final_bin["target_rate"] = final_bin["target_rate"].map(
                    "{:,.2%}".format
                )
                final_bin["dist_good"] = final_bin["dist_good"].map("{:,.2%}".format)
                final_bin["dist_bad"] = final_bin["dist_bad"].map("{:,.2%}".format)
                final_bin["woe"] = round(final_bin["woe"], 6)
                final_bin["iv"] = round(final_bin["iv"], 4)

                final_bin["value"] = final_bin.index
                final_bin = final_bin.reset_index()
                final_bin = final_bin[
                    [
                        "attr",
                        "value",
                        "#accts",
                        "%accts",
                        "#good",
                        "#bad",
                        "dist_good",
                        "dist_bad",
                        "target_rate",
                        "woe",
                        "iv",
                    ]
                ]

                final_bin.loc[final_bin["#bad"].isnull(), "woe"] = -5
                final_bin.loc[final_bin["#good"].isnull(), "woe"] = 5

                # if #other/missing< self.min_samples_leaf/2, then assign woe=0
                final_bin.loc[
                    final_bin["#accts"] < self.min_samples_leaf / 2, "woe"
                ] = 0

                # information value
                self.iv[char] = round(final_bin["iv"].sum(), 4)

                # save the woe binning information
                mapping = XX[[char, char + "_t"]].drop_duplicates(
                    keep="first", inplace=False
                )
                woe_result = mapping.merge(
                    final_bin[["value", "woe", "%accts"]],
                    how="left",
                    left_on=char + "_t",
                    right_on="value",
                )
                self.woe[char] = woe_result[[char, "woe", "%accts"]]
                self.iv_detail[char] = final_bin

                # print the results
                if i <= self.display:
                    print(Fore.BLUE + char, "-iv: ", self.iv[char], Style.RESET_ALL)
                    print(final_bin)
                    print()

        print("processed ", i, " char attributes ")
        self.char_list = list(set(self.char_list) - set(to_remove))
        return self

    # woe for numerical attributes
    def _bin_num(self, X, Y, Y_weight, speical_value=[]):

        # print("processing num attributes: ")
        XX = X.copy()
        kk = 0

        for name in self.num_list_local:

            kk = kk + 1
            self.cutoffs[name] = []

            YXX = pd.DataFrame()
            YXX = pd.concat([Y, Y_weight, XX[name]], axis=1)

            YXX.columns = ["target", "weight", name]

            YXX["target_weight"] = YXX["target"] * YXX["weight"]

            # seperate data to missing/special/normal
            YXX_missing = YXX[YXX[name].isnull() == True]
            YXX_special = YXX[YXX[name].isin(speical_value)]
            YXX_normal = YXX[
                (YXX[name].isin(speical_value) == False) & (YXX[name].isnull() == False)
            ]

            if len(YXX_normal) <= 500:
                # YXX_normal=YXX[(YXX[name].isnull()==False)]
                # YXX_special=pd.DataFrame()
                if self.display > 0:
                    print(name, ": not enought valid value")
                self.iv[name] = 0
                self.invalid_list.append(name)
                continue

            #  find cutoff by equal bin
            if self.method == "equal":
                n_uniq = YXX_normal[name].nunique()

                if n_uniq <= self.num_bins:
                    nn = n_uniq - 1
                else:
                    nn = self.num_bins
                list1 = pd.qcut(
                    YXX_normal[name], nn, retbins=True, labels=False, duplicates="drop"
                )[1].tolist()

                self.cutoffs[name] = list1[1 : len(list1) - 1]

            # find cutoff by tree
            if self.method == "tree":
                estimator = DecisionTreeClassifier(
                    min_samples_leaf=self.min_samples_leaf, max_leaf_nodes=self.num_bins
                )
                estimator.fit(
                    YXX_normal[[name]],
                    YXX_normal["target"],
                    sample_weight=YXX_normal["weight"].values,
                )
                dtree = estimator.tree_

                def tree_recurse(node):
                    if dtree.feature[node] != -2:
                        threshold = dtree.threshold[node]
                        if (
                            dtree.feature[dtree.children_left[node]] == -2
                            or dtree.feature[dtree.children_right[node]] == -2
                        ):
                            self.cutoffs[name].append(threshold)
                        tree_recurse(dtree.children_left[node])
                        tree_recurse(dtree.children_right[node])

                tree_recurse(0)
                self.cutoffs[name].sort()

            self.cutoffs[name].insert(len(self.cutoffs[name]), float("inf"))
            self.cutoffs[name].insert(0, float("-inf"))

            # converge conditon to bin to make sure monotone
            corr_n = 0
            start = 0
            while abs(round(corr_n, 4)) < 1:

                bin2 = pd.cut(
                    YXX_normal[name], self.cutoffs[name], right=True, labels=False
                )
                weight_bin = YXX_normal.groupby(bin2)["weight"].sum()
                target_rate = round(
                    YXX_normal.groupby(bin2)["target_weight"].sum()
                    / YXX_normal.groupby(bin2)["weight"].sum(),
                    4,
                )
                # print(target_rate)

                corr, p = stats.spearmanr(
                    YXX_normal.groupby(bin2)[name].mean(), target_rate
                )
                corr_w = wpearson(
                    YXX_normal.groupby(bin2)[name].mean(), target_rate, weight_bin
                )

                if corr * corr_w > 0:
                    corr_n = corr
                else:
                    corr_n = corr_w

                ## if weighted corr value is invalid
                if corr_w == 2:
                    corr_n = corr

                sign_corr = corr_n > 0

                pop = []
                i = start
                while i <= len(self.cutoffs[name]) - 3:

                    if i not in list(target_rate.index):
                        target_rate.reset_index(drop=True, inplace=True)

                    if i + 1 not in list(target_rate.index):
                        target_rate.reset_index(drop=True, inplace=True)
                        i = i - 1

                    sign_neighbor = (target_rate[i + 1] - target_rate[i]) > 0

                    if sign_neighbor != sign_corr:
                        pop.append(self.cutoffs[name][i + 1])

                    elif abs(target_rate[i + 1] - target_rate[i]) <= 0.001:
                        pop.append(self.cutoffs[name][i + 1])

                    i = i + 2

                self.cutoffs[name] = list(set(self.cutoffs[name]) - set(pop))
                self.cutoffs[name].sort()

                start = abs(1 - start)

            bin2 = pd.cut(
                YXX_normal[name], self.cutoffs[name], right=True, labels=False
            )
            min_v = YXX_normal.groupby(bin2)[name].min()
            max_v = YXX_normal.groupby(bin2)[name].max()
            num_accts = YXX_normal.groupby(bin2)["weight"].sum()
            num_bad = round(
                YXX_normal[YXX_normal["target"] == 1].groupby(bin2)["weight"].sum(), 1
            )
            num_good = round(
                YXX_normal[YXX_normal["target"] == 0].groupby(bin2)["weight"].sum(), 1
            )

            # num_bad=0 if num_bad is None
            # num_good=0 if num_good is None

            target_rate = num_bad / num_accts
            normal = pd.concat(
                [min_v, max_v, num_accts, num_good, num_bad, target_rate], axis=1
            )
            normal.columns = ["min", "max", "#accts", "#good", "#bad", "target_rate"]
            normal.reset_index(drop=True, inplace=True)

            # for missing part
            missing = pd.DataFrame()
            if YXX_missing.empty == False:
                m_num_accts = YXX_missing["weight"].sum()
                m_num_bad = round(
                    YXX_missing[YXX_missing["target"] == 1]["weight"].sum(), 1
                )
                m_num_good = round(
                    YXX_missing[YXX_missing["target"] == 0]["weight"].sum(), 1
                )

                # m_num_bad=0 if m_num_bad is None
                # m_num_good=0 if m_num_good is None

                m_target_rate = m_num_bad / m_num_accts
                missing = pd.DataFrame(
                    np.array(
                        [
                            [
                                np.nan,
                                np.nan,
                                m_num_accts,
                                m_num_good,
                                m_num_bad,
                                m_target_rate,
                            ]
                        ]
                    ),
                    columns=["min", "max", "#accts", "#good", "#bad", "target_rate"],
                )
                missing.index = ["missing"]

            # for special part
            special = pd.DataFrame()
            if YXX_special.empty == False:
                s_min = YXX_special[name].min()
                s_max = YXX_special[name].max()
                s_num_accts = YXX_special["weight"].sum()
                s_num_bad = round(
                    YXX_special[YXX_special["target"] == 1]["weight"].sum(), 1
                )
                s_num_good = round(
                    YXX_special[YXX_special["target"] == 0]["weight"].sum(), 1
                )

                # s_num_bad=0 if s_num_bad is None
                # s_num_good=0 if s_num_good is None

                s_target_rate = s_num_bad / s_num_accts
                special = pd.DataFrame(
                    np.array(
                        [
                            [
                                s_min,
                                s_max,
                                s_num_accts,
                                s_num_good,
                                s_num_bad,
                                s_target_rate,
                            ]
                        ]
                    ),
                    columns=["min", "max", "#accts", "#good", "#bad", "target_rate"],
                )
                special.index = ["special"]

            # combine three part together
            # calculate woe and iv
            final_bin = pd.concat([normal, special, missing])

            # del final_bin.index.name

            final_bin["%accts"] = final_bin["#accts"] / final_bin["#accts"].sum()
            final_bin["dist_good"] = final_bin["#good"] / final_bin["#good"].sum()
            final_bin["dist_bad"] = final_bin["#bad"] / final_bin["#bad"].sum()

            final_bin["woe"] = np.log(final_bin["dist_bad"] / final_bin["dist_good"])
            final_bin["iv"] = (
                final_bin["dist_bad"] - final_bin["dist_good"]
            ) * final_bin["woe"]

            final_bin["attr"] = name
            final_bin["%accts"] = final_bin["%accts"].map("{:,.2%}".format)
            final_bin["target_rate"] = final_bin["target_rate"].map("{:,.2%}".format)
            final_bin["dist_good"] = final_bin["dist_good"].map("{:,.2%}".format)
            final_bin["dist_bad"] = final_bin["dist_bad"].map("{:,.2%}".format)
            final_bin["woe"] = round(final_bin["woe"], 4)
            final_bin["iv"] = round(final_bin["iv"], 4)
            final_bin["min"] = round(final_bin["min"], 4)
            final_bin["max"] = round(final_bin["max"], 4)

            final_bin = final_bin[
                [
                    "attr",
                    "min",
                    "max",
                    "#accts",
                    "%accts",
                    "#good",
                    "#bad",
                    "dist_good",
                    "dist_bad",
                    "target_rate",
                    "woe",
                    "iv",
                ]
            ]

            final_bin.loc[final_bin["#bad"].isnull(), "woe"] = -5
            final_bin.loc[final_bin["#good"].isnull(), "woe"] = 5

            # final_bin.loc[final_bin['#accts']<self.min_samples_leaf/2,'woe']=0
            final_bin.loc[final_bin["#accts"] < self.min_samples_leaf / 2, "iv"] = 0

            self.iv[name] = round(final_bin["iv"].sum(), 4)
            self.woe[name] = final_bin[["%accts", "min", "max", "woe"]]
            self.special[name] = speical_value
            self.iv_detail[name] = final_bin

            if YXX_special.empty:
                self.special[name] = []

            if kk <= self.display:
                print(Fore.BLUE + name, "-iv: ", self.iv[name], Style.RESET_ALL)
                print(final_bin)
                print()
        print("processed ", kk, " num attributes")
        print()
        return self

    # fit the raw atttributes
    def fit(self, X, Y, Y_weight=pd.Series(), display=10, special_value=[], sort="Y"):
        self.display = display
        self._check(X, Y)

        if Y_weight.empty:
            Y_weight = pd.Series(np.ones(len(Y)))
            Y_weight.index = Y.index
        elif len(Y_weight) != len(Y_weight) or Y_weight.isnull().sum() > 0:
            raise ValueError("weight does not match with target")

        if len(self.char_list_local) > 0:
            self._bin_char(X, Y, Y_weight)

        self._bin_num(X, Y, Y_weight, special_value)

    # return the information value
    def get_iv(self):
        iv = pd.DataFrame(list(self.iv.items()))
        iv.columns = ["attr", "iv"]
        return iv

    # transform data
    def transform(self, X, train_data=0, keep=False):

        XX = X.copy()
        XX["id"] = XX.index
        self._check_transform(XX)

        iv = self.get_iv()
        iv_remove = iv.loc[iv["iv"] <= self.min_iv, "attr"].to_list()
        if iv_remove:
            print(
                len(iv_remove),
                " attributes below will be removed form transfomration beacause their information value is below threshold:",
            )
            print(iv_remove)
            print()
            print()

        num_list = list(set(self.num_list) - set(iv_remove))
        char_list = list(set(self.char_list) - set(iv_remove))

        transformed = []

        char_i = 0

        for char in char_list:
            char_i = char_i + 1
            if char_i % 20 == 0:
                print("transformed", char_i)
            XX = XX.merge(self.woe[char], how="left", left_on=char, right_on=char)
            XX.loc[XX["woe"].isnull(), "woe"] = 0
            XX.rename(columns={"woe": char + "_woe"}, inplace=True)
            transformed.append(char + "_woe")

            if XX[char + "_woe"].isnull().sum() > 0:
                raise ValueError(char, "woe transfomr fails")

        if char_i > 0:
            print("transformed", char_i)
        # print()

        num_transformed = 0

        for num in num_list:
            # print(num)

            num_transformed = num_transformed + 1
            if num_transformed % 20 == 0:
                print("transformed num", num_transformed)

            woe_value = self.woe[num].reset_index()
            woe_value_list = woe_value[
                woe_value["index"].isin(["special", "missing"]) == False
            ]["woe"].to_list()

            if len(self.cutoffs[num]) == len(woe_value_list) + 1:
                cutoff_list = self.cutoffs[num]
            else:
                cutoff_list = (
                    [float("-inf")]
                    + woe_value[
                        woe_value["index"].isin(["special", "missing"]) == False
                    ]["max"].to_list()[:-1]
                    + [float("inf")]
                )

            XX[num + "_woe"] = pd.cut(
                XX[num], cutoff_list, right=True, labels=woe_value_list
            )
            XX[num + "_woe"] = XX[num + "_woe"].astype(float)

            XX.loc[XX[num].isin(self.special[num]), num + "_woe"] = woe_value[
                woe_value["index"] == "special"
            ]["woe"].values

            if len(woe_value[woe_value["index"] == "missing"]) > 0:
                XX.loc[XX[num].isnull(), num + "_woe"] = woe_value[
                    woe_value["index"] == "missing"
                ]["woe"].values.item()
            else:
                XX.loc[XX[num].isnull(), num + "_woe"] = 0

            transformed.append(num + "_woe")

            if XX[num + "_woe"].isnull().sum() > 0:
                raise ValueError(num, "woe transfomr fails")

        # print('transformed',num_transformed)
        print()
        XX.set_index("id", inplace=True)
        XX.index.name = None

        self.psi = {}
        self.psi_detail = {}

        # calculate psi
        if train_data == 0:
            for name in num_list:
                t_dist_min = XX.groupby(name + "_woe")[name].min()
                t_dist_max = XX.groupby(name + "_woe")[name].max()
                t_dist_count = XX.groupby(name + "_woe").size()
                t_dist_pct = XX.groupby(name + "_woe").size() / len(XX)
                t_dist_df = pd.concat(
                    [t_dist_min, t_dist_max, t_dist_count, t_dist_pct], axis=1
                )
                t_dist_df.reset_index(inplace=True)
                t_dist_df.columns = ["woe", "new_min", "new_max", "#accts", "new_dist"]
                t_dist_df["new_dist"] = round(t_dist_df["new_dist"], 4)

                t_dist_pct_new = t_dist_df.merge(
                    self.woe[name], how="outer", left_on="woe", right_on="woe"
                )
                t_dist_pct_new["%accts"] = (
                    t_dist_pct_new["%accts"].str.rstrip("%").astype("float") / 100.0
                )
                t_dist_pct_new.loc[
                    t_dist_pct_new["new_dist"].isnull(), "new_dist"
                ] = 0.0001
                t_dist_pct_new.loc[t_dist_pct_new["%accts"] == 0, "%accts"] = 0.0001
                t_dist_pct_new["psi"] = round(
                    (t_dist_pct_new["new_dist"] - t_dist_pct_new["%accts"])
                    * np.log(t_dist_pct_new["new_dist"] / t_dist_pct_new["%accts"]),
                    5,
                )
                t_dist_pct_new.rename(columns={"%accts": "orig_dist"}, inplace=True)
                t_dist_pct_new["orig_dist"] = t_dist_pct_new["orig_dist"].map(
                    "{:,.2%}".format
                )
                t_dist_pct_new["new_dist"] = t_dist_pct_new["new_dist"].map(
                    "{:,.2%}".format
                )
                t_dist_pct_new = (
                    t_dist_pct_new[
                        ["min", "max", "orig_dist", "new_dist", "#accts", "psi"]
                    ]
                    .sort_values(by="min")
                    .reset_index()
                    .drop(columns="index")
                )

                self.psi[name] = round(t_dist_pct_new["psi"].sum(), 5)
                self.psi_detail[name] = t_dist_pct_new

            for name in char_list:
                t_dist_pct = pd.DataFrame(
                    XX.groupby(name + "_woe")[name].count() / len(XX)
                )
                t_dist_pct.reset_index(inplace=True)
                t_dist_pct.columns = ["woe", "new_dist"]
                t_dist_pct["new_dist"] = round(t_dist_pct["new_dist"], 4)
                woe_char = self.woe[name]
                woe_char["dup"] = woe_char.duplicated("woe")
                woe_char["dup_2"] = woe_char.duplicated("woe", keep=False)
                woe_char_new = woe_char[woe_char["dup"] == False]
                woe_char_new.loc[woe_char_new["dup_2"] == True, name] = "0: other"
                woe_char_new.drop(columns=["dup", "dup_2"], inplace=True)
                t_dist_pct_new = t_dist_pct.merge(
                    woe_char_new, how="outer", left_on="woe", right_on="woe"
                )

                t_dist_pct_new["%accts"] = (
                    t_dist_pct_new["%accts"].str.rstrip("%").astype("float") / 100.0
                )
                t_dist_pct_new.loc[
                    t_dist_pct_new["new_dist"].isnull(), "new_dist"
                ] = 0.0001
                t_dist_pct_new.loc[t_dist_pct_new["%accts"] == 0, "%accts"] = 0.0001

                t_dist_pct_new["psi"] = round(
                    (t_dist_pct_new["new_dist"] - t_dist_pct_new["%accts"])
                    * np.log(t_dist_pct_new["new_dist"] / t_dist_pct_new["%accts"]),
                    5,
                )
                t_dist_pct_new.rename(columns={"%accts": "orig_dist"}, inplace=True)
                self.psi[name] = round(t_dist_pct_new["psi"].sum(), 5)
                self.psi_detail[name] = (
                    t_dist_pct_new[[name, "woe", "orig_dist", "new_dist", "psi"]]
                    .sort_values(by="orig_dist")
                    .reset_index()
                )

        if keep == False:
            return XX[transformed]

        else:
            return XX

    # show psi
    def show_psi(self, detail=1, attrs_list=None):

        if attr_list:
            for name in attr_list:
                print(Fore.BLUE + name, "-psi: ", self.psi[name], Style.RESET_ALL)
                if detail == 1:
                    print(self.psi_detail[name])
                    print()
        else:
            for name in self.psi:
                print(Fore.BLUE + name, "-psi: ", self.psi[name], Style.RESET_ALL)
                if detail == 1:
                    print(self.psi_detail[name])
                    print()

    def display_bin_results(self, attrs_list=None, simple=1, out=None):
        if attrs_list is None:
            list1 = self.num_list
            for attr in self.invalid_list:
                list1.remove(attr)
        else:
            list1 = attrs_list

        if simple != 1:
            output_detail_list = [
                "attr",
                "min",
                "max",
                "#accts",
                "%accts",
                "#good",
                "#bad",
                "dist_good",
                "dist_bad",
                "target_rate",
                "woe",
                "iv",
            ]
        else:
            output_detail_list = [
                "attr",
                "min",
                "max",
                "#accts",
                "%accts",
                "#bad",
                "target_rate",
            ]

        df_out = pd.DataFrame(columns=output_detail_list)
        df_e = pd.DataFrame(columns=output_detail_list)
        df_e.loc["x"] = np.repeat(np.nan, len(output_detail_list))

        for attr in list1:
            df_out = df_out.append(self.iv_detail[attr][output_detail_list])
            df_out = df_out.append(df_e)
            if out is None:
                print(self.iv_detail[attr][output_detail_list])
                print()
        if out:
            df_out = df_out[output_detail_list]
            df_out.to_csv(out)

    def woe_dict(self):
        return self.woe

    ### generate json file for woe
    def woe_json(self, attrs_list):
        woe_json = {}
        for attr in attrs_list:

            if attr in self.num_list:
                woe_json[attr] = {}
                woe_json[attr]["type"] = "num"
                woe_json[attr]["cutoff"] = self.cutoffs[attr]
                woe_json[attr]["special"] = self.special[attr]

                woe_json[attr]["index"] = self.woe[attr].index.to_list()
                woe_json[attr]["woe"] = self.woe[attr]["woe"].to_list()
                woe_json[attr]["min"] = self.woe[attr]["min"].to_list()
                woe_json[attr]["max"] = self.woe[attr]["max"].to_list()

            elif attr in self.char_list:
                woe_json[attr] = {}
                woe_json[attr]["type"] = "char"

                woe_json[attr]["index"] = self.woe[attr].index.to_list()
                woe_json[attr]["value"] = self.woe[attr][attr].to_list()
                woe_json[attr]["woe"] = self.woe[attr]["woe"].to_list()

            else:
                raise ValueError("attributes not found: ", attr)

        outjson = json.dumps(woe_json)
        return outjson

    # transform data
    def transform_from_json(self, X, json_file, attr_list=None, keep=False):

        XX = X.copy()
        XX["id"] = XX.index
        dict_woe = json.loads(json_file)
        woe_list = list(dict_woe.keys())

        if attr_list == None:
            final_list = woe_list

        elif set(attr_list).issubset(set(woe_list)) == False:
            print("mismatch,please check attr_list")
        else:
            final_list = attr_list

        transformed = []
        for attr in final_list:
            # print(attr)
            if dict_woe[attr]["type"] == "char":
                dict_woe_attr = dict_woe[attr].copy()
                dict_woe_attr.pop("type")
                df_woe = pd.DataFrame.from_dict(dict_woe_attr).set_index("index")
                XX = XX.merge(df_woe, how="left", left_on=attr, right_on="value")
                XX.loc[XX["woe"].isnull(), "woe"] = 0
                XX.rename(columns={"woe": attr + "_woe"}, inplace=True)
                if XX[attr + "_woe"].isnull().sum() > 0:
                    raise ValueError(attr, "woe transfomr fails")

            if dict_woe[attr]["type"] == "num":

                dict_woe_attr = dict_woe[attr].copy()

                dict_woe_attr.pop("type")
                dict_woe_attr.pop("cutoff")
                dict_woe_attr.pop("special")

                woe_value = pd.DataFrame.from_dict(dict_woe_attr)
                woe_value_list = woe_value[
                    woe_value["index"].isin(["special", "missing"]) == False
                ]["woe"].to_list()

                if len(dict_woe[attr]["cutoff"]) == len(woe_value_list) + 1:
                    cutoff_list = dict_woe[attr]["cutoff"]
                else:
                    cutoff_list = (
                        [float("-inf")]
                        + woe_value[
                            woe_value["index"].isin(["special", "missing"]) == False
                        ]["max"].to_list()[:-1]
                        + [float("inf")]
                    )

                XX[attr + "_woe"] = pd.cut(
                    XX[attr], cutoff_list, right=True, labels=woe_value_list
                )
                XX[attr + "_woe"] = XX[attr + "_woe"].astype(float)

                XX.loc[
                    XX[attr].isin(dict_woe[attr]["special"]), attr + "_woe"
                ] = woe_value[woe_value["index"] == "special"]["woe"].values

                if len(woe_value[woe_value["index"] == "missing"]) > 0:
                    XX.loc[XX[attr].isnull(), attr + "_woe"] = woe_value[
                        woe_value["index"] == "missing"
                    ]["woe"].values
                else:
                    XX.loc[XX[attr].isnull(), attr + "_woe"] = 0

                if XX[attr + "_woe"].isnull().sum() > 0:
                    raise ValueError(num, "woe transfomr fails")

            transformed.append(attr + "_woe")

        print("# Transformed:", len(transformed))

        XX.set_index("id", inplace=True)
        XX.index.name = None

        if keep == False:
            return XX[transformed]
        else:
            return XX
