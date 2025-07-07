import sys, os, json
import numpy as np
import pandas as pd


####################################
#             config
####################################


def get_default_config(to_json_path=None):
    """
    produce default config file

    each config contains 5 subfields:
    data: shared data configuration and paths
    train: objects for training
    monitor: objects for monitoring
    meta: meta needed/produced during modeling
    other: anything that does not fit in previous segments
    """

    config = {}

    config["data"] = {}  # shared data config
    config["train"] = {}  # training specific
    config["monitor"] = {}  # monitoring specific
    config["meta"] = {}  # meta data specific
    config["other"] = {}  # miscellaneous

    for key in ["raw", "processed", "joined", "features", "labeled"]:
        config["data"][key] = {}

    config["train"]["features"] = []
    config["train"]["model_params"] = {}
    config["train"]["train_data_path"] = None  # single data file
    config["train"][
        "valid_data_paths"
    ] = {}  # dictionary: key = file_name, value: directory
    config["train"][
        "test_data_paths"
    ] = {}  # dictionary: key = file_name, value: directory
    config["train"]["baseline_models"] = {}

    config["monitor"]["dev_data_paths"] = {}  # data in the dev environment
    config["monitor"]["prod_data_paths"] = {}  # data in the prod enviornment
    config["monitor"]["baseline_models"] = {}

    config["meta"]["target_col"] = None
    config["meta"]["indeterminate_col"] = None
    config["meta"]["meta_cols"] = []
    config["meta"]["time_id"] = []

    if isinstance(to_json_path, str):
        with open(to_json_path, "w") as f:
            json.dump(config, f, indent=4)

    return config


def load_config(path):
    """helper to load config file"""
    with open(path, "r") as f:
        return json.load(f)


def to_json(json_object, path):
    """helper to write to json object"""
    with open(path, "w") as f:
        json.dump(json_object, f, indent=4)


####################################
#             datetime
####################################


def monthdelta(date, delta):
    """
    get datetime by shifting date by delta

    e.g.
        import pandas as pd
        monthdelta(pd.to_datetime("2021-03-13"), -6)

        >>> Timestamp('2020-09-13 00:00:00')
    """
    m, y = (date.month + delta) % 12, date.year + ((date.month) + delta - 1) // 12
    if not m:
        m = 12
    d = min(
        date.day,
        [
            31,
            29 if y % 4 == 0 and (not y % 100 == 0 or y % 400 == 0) else 28,
            31,
            30,
            31,
            30,
            31,
            31,
            30,
            31,
            30,
            31,
        ][m - 1],
    )
    return date.replace(day=d, month=m, year=y)


####################################
#              aws
####################################


def read_pickle_from_s3(bucket, key):
    import pickle
    import boto3

    s3 = boto3.resource("s3")
    my_pickle = pickle.loads(s3.Bucket(bucket).Object(key).get()["Body"].read())
    return my_pickle


def write_pickle_to_s3(obj, bucket, key):
    import io
    import boto3
    import pickle as pkl

    s3_resource = boto3.resource("s3")
    obj_pkl = pkl.dumps(obj)
    s3_resource.Object(bucket, key).put(Body=obj_pkl)


####################################
#        Generate fake data
####################################


def create_fake_pii(num=1, return_df=True):
    from faker import Faker
    import random

    fake = Faker()
    output = [
        {
            "name": fake.name(),
            "address": fake.address(),
            "name": fake.name(),
            "email": fake.email(),
            "bs": fake.bs(),
            "address": fake.address(),
            "city": fake.city(),
            "state": fake.state(),
            "date_time": fake.date_time(),
            "paragraph": fake.paragraph(),
            "Conrad": fake.catch_phrase(),
            "randomdata": random.randint(1000, 2000),
        }
        for x in range(num)
    ]
    return output if not return_df else pd.DataFrame(output)


def create_fake_data(num=1, return_df=True):
    from faker import Faker
    import random

    fake = Faker()
    output = [
        {
            "feature1": random.random() * 10,
            "feature2": random.random() * 5,
            "feature3": random.random() * 100,
            "feature4": random.random() * 2,
            "feature5": random.randint(0, 10),
            "id": x,
            "target": random.randint(0, 1),
        }
        for x in range(num)
    ]
    return output if not return_df else pd.DataFrame(output)
