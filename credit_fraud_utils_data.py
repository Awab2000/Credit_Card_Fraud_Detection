import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from collections import Counter

def load_data(dataset):
    df = pd.read_csv(dataset)
    return df


def remove_duplicates(data):
    data = data.loc[~data.duplicated()].reset_index(drop=True).copy()
    return data


def from_df_to_np(df):
    y = df['Class'].to_numpy()
    X = df.drop(['Class'], axis=1).to_numpy()
    return X, y


def transform_train_val(X_train, X_val, option):
    if option == 0:
        return X_train, X_val, None
    elif option == 1:
        processor = MinMaxScaler()
    else:
        processor = StandardScaler()
    X_train = processor.fit_transform(X_train)
    if X_val is not None:
        X_val = processor.transform(X_val)
    return X_train, X_val, processor


def poly_process(X_train, X_val, degree):
    poly = PolynomialFeatures(degree=degree, interaction_only=False, include_bias=False)
    X_train = poly.fit_transform(X_train)
    if X_val is not None:
        X_val = poly.fit_transform(X_val)
    return X_train, X_val


def log_transform(X_train, X_val = None, X_test=None):
    X_train = np.log1p(np.maximum(X_train, 0))
    if X_val is not None:
        X_val = np.log1p(np.maximum(X_val, 0))
    if X_test is not None:
        X_test = np.log1p(np.maximum(X_test, 0))
    return X_train, X_val, X_test


def under_sample(X, y):
    counter = Counter(y)
    print(counter)
    factor, minority_size = 45, counter[1]
    rus = RandomUnderSampler(sampling_strategy={0: factor * minority_size})
    X_us, y_us = rus.fit_resample(X, y)
    print(Counter(y_us))

    return X_us, y_us


def over_sample(X, y):
    counter = Counter(y)
    print(counter)
    factor, majority_size = 300, counter[0]
    new_sz = int(majority_size / factor)
    oversample = SMOTE(random_state=1, sampling_strategy={1: new_sz}, k_neighbors=3)
    X_os, y_os = oversample.fit_resample(X, y)
    counter = Counter(y_os)
    print(counter)

    return X_os, y_os


def under_over_sample(X, y):
    counter = Counter(y)
    print(counter)
    factor1, minority_size = 100, counter[1]
    factor2, majority_size = 100, counter[0]
    new_maj_sz = minority_size * factor1
    new_min_sz = int(majority_size / factor2)

    undersample = RandomUnderSampler(random_state=1, sampling_strategy={0: new_maj_sz})
    oversample = SMOTE(random_state=1, sampling_strategy={1: new_min_sz}, k_neighbors=3)

    from imblearn.pipeline import Pipeline as imb_Pipeline
    pip = imb_Pipeline(steps=[('over', oversample), ('under', undersample)])

    X_ous, y_ous = pip.fit_resample(X, y)
    counter = Counter(y_ous)
    print(counter)

    return X_ous, y_ous