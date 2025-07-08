import sys, os, json
import numpy as np
import pandas as pd


class MonitorBase(object):
    def __init__(self, artifact_path: str, context: dict = {}):
        """
        Provide initial context that will passed down to the modules

        Please overwrite the following methods:
            get_dev_data
            get_prod_data
            get_pred
            get_performance_report
            refit

        And self.run(context) will call the 5 functions in the presented order.

        @params context: dict
            dictionary that contains from init and upstream stages
        @returns context: dict
            everything needed for downstream tasks
        """
        self.artifact_path = artifact_path
        self.context = context

    def get_dev_data(self, context: dict):
        """
        Get development data. Please overwrite for specific tasks

        Please complete all data wrangling for data from the dev environment here

        @params context: dict
            dictionary that contains from init and upstream stages
        @returns context: dict
            everything needed for the downstream task of make_pred
        """
        raise NotImplemented("Please overwrite this method")

    def get_prod_data(self, context: dict):
        """
        Get production data. Please overwrite for specific tasks

        Please complete all data wrangling for data from the prod environment here

        @params context: dict
            dictionary that contains from init and upstream stages
        @returns context: dict
            everything needed for the downstream task of make_pred
        """
        raise NotImplemented("Please overwrite this method")

    def get_pred(self, context: dict):
        """
        Make prediction on both dev and prod data.
        Please overwrite for specific tasks.

        @params context: dict
            dictionary that contains from init and upstream stages
        @returns context: dict
            everything needed for the downstream task of make_pred
        """
        raise NotImplemented("Please overwrite this method")

    def get_performance_report(self, context: dict):
        """
        Produce performance reports using context.
        Please overwrite for specific tasks

        @params context: dict
            dictionary that contains from init and upstream stages
        @returns context: dict
            everything needed for the downstream task of make_pred
        """
        raise NotImplemented("Please overwrite this method")

    def refit(self, context: dict):
        """
        Refitting step, Please overwrite for specific tasks.

        @params context: dict
            dictionary that contains from init and upstream stages
        @returns context: dict
            everything needed for the downstream task of make_pred
        """
        raise NotImplemented

    def get_context(self):
        import copy

        return copy.deepcopy(self.context)

    def run(self):
        """
        run the following tasks in the provided order, with the context

        provided at the initialization stage.
            get_dev_data
            get_prod_data
            get_pred
            get_performance_report
            refit

        @params: None
        @return: None
        """
        self.context = self.get_dev_data(self.context)
        self.context = self.get_prod_data(self.context)
        self.context = self.get_pred(self.context)
        self.context = self.get_performance_report(self.context)
        self.context = self.refit(self.context)
