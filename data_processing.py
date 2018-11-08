import pandas as pd
import numpy as np
import pickle as pkl

from settings import *

def convert_to_datetime(df, column_name):
    df[column_name] = pd.to_datetime(df[column_name])

# converts to NDF and not NDF labels
def convert_to_labels(df, label_column, name):
    df[name] = (train_users[label_column] != 'NDF').astype(int)

def to_dummy(df, column_name, prefix):
    new_col = pd.get_dummies(df[column_name], prefix=prefix)
    df[new_col.columns] = new_col

def to_categorical(df, column_name, prefix):
    df[prefix + column_name] = df[column_name].astype('category').cat.codes

def get_year(date):
    return date.year

def get_month(date):
    return date.month

def get_day(date):
    return date.day

def drop_data_columns(df):
    return df.drop(columns=drop_columns)

def agg_sessions(sess_df):
    return sess_df.groupby('user_id', as_index=False)['secs_elapsed'].sum()

def merge_df(left_df, right_df, left_column, right_column, how='inner'):
    result = pd.merge(left_df, right_df, left_on=left_column, right_on=right_column, how=how)
    result = result.drop(columns=right_column)
    return result

def extract_date_features(df, column_name):
    df[column_name + '_year'] = df[column_name].apply(get_year)
    df[column_name + '_month'] = df[column_name].apply(get_month)
    df[column_name + '_day'] = df[column_name].apply(get_day)

    df[column_name + '_year'].fillna(0, inplace=True)
    df[column_name + '_month'].fillna(0, inplace=True)
    df[column_name + '_day'].fillna(0, inplace=True)


def process_session_sec(df, column_name, prefix, median_sec=None):
    new_column_name = prefix + column_name
    df[new_column_name] = df[column_name]
    
    if not median_sec:
        median_sec = df[new_column_name].median(skipna=True)
        print("seconds_median: {}".format(median_sec))

    df[new_column_name].fillna(median_sec, inplace=True)
    assert(df[new_column_name].isna().sum()==0)

    return median_sec

def process_age(df, column_name, prefix, median_age=None):
    new_column_name = prefix + column_name
    df[new_column_name] = df[column_name]
    column_data = np.array(df[new_column_name].values.tolist())
    df[new_column_name] = np.where(column_data > 115, np.NaN, column_data).tolist()
    
    if not median_age:
        median_age = df[new_column_name].median()
        print("median age: {}".format(median_age))

    df[new_column_name].fillna(median_age, inplace=True)
    assert(df[new_column_name].isna().sum() == 0)

    return median_age


def process_labels(df, column_name, prefix):
    new_column_name = prefix  + column_name
    df[new_column_name] = (df[column_name] != 'NDF').astype(int)


def save_metadata(median_sec, median_age, df_feature_columns):
    meta_data = {}
    meta_data['median_sec'] = median_sec
    meta_data['median_age'] = median_age
    meta_data['feature_columns'] = df_feature_columns

    with open(METADATA_PATH, 'wb') as file:
        pkl.dump(meta_data, file)


def load_metadata():
    with open(METADATA_PATH, 'rb') as file:
        meta_data = pkl.load(file)

    return meta_data

def preprocess_df(df, train=True):
    # check validity in test set?
    # extract_date_features(df, 'date_first_booking')

    # read these values from a file for test

    median_age = None
    median_sec = None
    df_feature_columns = None
    if not train:
        meta_data = load_metadata()
        median_age = meta_data['median_age']
        median_sec = meta_data['median_sec']
        df_feature_columns = meta_data['feature_columns']

    # print("Median age {} Median sec {}".format(median_age, median_sec))

    process_labels(df, 'country_destination', 'label_')
    
    median_sec = process_session_sec(df, 'secs_elapsed', 'processed_', median_sec=median_sec)
    median_age = process_age(df, 'age', 'processed_', median_age=median_age)

    to_categorical(df, 'signup_flow', 'processed_')
    to_categorical(df, 'first_browser', 'processed_')
    to_dummy(df, 'gender', 'gender')
    to_dummy(df, 'first_device_type', 'first_device_type')
    to_dummy(df, 'signup_app', 'signup_type')
    to_dummy(df, 'signup_method', 'signup_method')
    to_dummy(df, 'language', 'language_used')
    to_dummy(df, 'affiliate_channel', 'affiliate_channel')
    to_dummy(df, 'signup_app', 'signup_app')
    to_dummy(df, 'first_affiliate_tracked', 'first_aff_tracked')

    if train:
        df = df.drop(columns=DROP_COLUMNS)
        df_feature_columns = list(df.columns.values)
        df_feature_columns.remove(LABEL_COLUMN)
        save_metadata(median_sec, median_age, df_feature_columns)    

    
    # for test df_feature columns should be read from a file
    if DEBUG:
        print("NUM COLUMNS: {}".format(df.columns.values.shape))
        print("Test Columns {}".format(df.columns))
        df_debug = df[df_feature_columns]
        print("Final columns {}".format(df_debug.columns))
    
    
    x_features = df[df_feature_columns].values
    y_labels = df[[LABEL_COLUMN]].values

    return x_features, y_labels.flatten()