import sys
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod


class DataDictionaryBase(metaclass=ABCMeta):
    def __init__(
        self,
        df,
        name_col,
        type_col,
        description_col,
        valid_val_col,
        default_exclusions_col,
    ):
        self.dd = df
        self.name_col = name_col
        self.type_col = type_col
        self.description_col = description_col
        self.default_exclusions_col = default_exclusions_col
        self.valid_val_col = valid_val_col

    @abstractmethod
    def process_default_exclusions(self, series):
        pass

    @abstractmethod
    def process_numerical(self, series):
        pass

    @abstractmethod
    def process_data_type(self, series):
        pass


class ExperianDataDict(DataDictionaryBase):
    def __init__(
        self,
        df,
        name_col,
        type_col,
        description_col,
        valid_val_col,
        default_exclusions_col,
    ):
        super().__init__(
            df,
            name_col=name_col,
            type_col=type_col,
            description_col=description_col,
            default_exclusions_col=default_exclusions_col,
            valid_val_col=valid_val_col,
        )

        self.dd = self.process_dd(df)

    def process_default_exclusions(self, series, dtype):
        def cat_parser(x):
            x = str(x).replace("[", "").replace("]", "").replace("'", "")
            L = x.split("\n")
            L = [l.strip().split(":")[0] for l in L]
            L = [int(l) for l in L if l.isnumeric()]
            if len(L) == 0:
                return None
            return L

        return series.apply(cat_parser)

    def process_numerical(self, series, dtype):
        df_ = series.to_frame()
        df_.columns = ["valid_val"]
        df_["type"] = dtype

        def get_min(row):
            try:
                if str(row["valid_val"]).lower() in ["nan", "none"]:
                    return np.nan
                if (row["type"] == "int") and isinstance(row["valid_val"], str):
                    min_max = row["valid_val"].split("-")
                    if len(min_max) != 2:
                        return "MANUAL TREATMENT NEEDED"
                    return int(min_max[0].strip())
            except:
                return "MANUAL TREATMENT NEEDED"
            return "MANUAL TREATMENT NEEDED"

        def get_max(row):
            try:
                if str(row["valid_val"]).lower() in ["nan", "none"]:
                    return np.nan
                if (row["type"] == "int") and isinstance(row["valid_val"], str):
                    min_max = row["valid_val"].split("-")
                    if len(min_max) != 2:
                        return "MANUAL TREATMENT NEEDED"
                    return int(min_max[1].strip())
            except:
                return "MANUAL TREATMENT NEEDED"
            return "MANUAL TREATMENT NEEDED"

        df_["min"] = df_.apply(get_min, axis=1)
        df_["max"] = df_.apply(get_max, axis=1)
        return df_["min"], df_["max"]

    def process_data_type(self, series):
        from pandas._libs.tslibs.timestamps import Timestamp

        types = series.astype(str)
        mapper = {"int": "int", "varchar": "str", "bigint": "int", "date": "date"}
        types.replace(mapper, inplace=True)
        return types

    def process_dd(self, df):
        dd = df.copy()
        dd["type"] = self.process_data_type(dd[self.type_col])
        dd["categorical"] = self.process_default_exclusions(
            dd[self.default_exclusions_col], dd["type"]
        )
        dd["min"], dd["max"] = self.process_numerical(
            dd[self.valid_val_col], dd["type"]
        )
        return dd

    @property
    def data_dict(self):
        return self.dd
