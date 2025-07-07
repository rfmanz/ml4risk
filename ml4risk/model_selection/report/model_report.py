import numpy as np
import pandas as pd


class ModelReport(object):
    """
    Base class for generating model reports
    """

    def __init__(
        self, df: pd.DataFrame, artifact_path: str = None, context: dict = {}, **kwargs
    ):
        self.df = df
        self.artifact_path = artifact_path
        self.context = context
        for k, v in kwargs:
            setattr(self, k, v)

    def get_pred_reports(self, context: dict, **kwargs):
        raise NotImplemented

    def get_segmented_perf(self, context: dict, **kwargs):
        raise NotImplemented

    def get_model_vs_baseline(self, context: dict, **kwargs):
        raise NotImplemented

    def get_plots(self, context: dict, **kwargs):
        raise NotImplemented

    def run(self):
        """
        run all methods to generate reports
        please daisy chain your functions

        @params None
        @returns self.context
        """
        # self.context = self.get_pred_reports(self.context)
        # self.context = self.get_segmented_perf(self.context)
        # self.context = self.get_model_vs_baseline(self.context)
        # self.context = self.get_plots(self.context)
        raise NotImplemented
        return self.context
