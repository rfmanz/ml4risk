import pandas as pd

num_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
cat_features = ['Sex', 'Ticket', 'Cabin', 'Embarked']
features = num_features + cat_features
cat_idx = [5, 6, 7, 8]
target = ['Survived']

def categorical_encoder(df, cols):
    for col in cols:
        cats = df[col].unique()
        map_dict = dict(zip(cats, range(len(cats))))
        df[col] = df[col].map(map_dict)
    return df
