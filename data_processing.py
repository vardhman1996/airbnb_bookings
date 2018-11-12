import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.feature_selection import chi2
from settings import *

def convert_to_datetime(df, column_name):
    df[column_name] = pd.to_datetime(df[column_name])

# converts to NDF and not NDF labels
def convert_to_labels(df, label_column, name):
    df[name] = (train_users[label_column] != 'NDF').astype(int)

def to_dummy(df, column_name):
    new_col = pd.get_dummies(df[column_name], prefix=column_name)
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

# def extract_date_features(df, column_name):
#     df[column_name + '_year'] = df[column_name].apply(get_year)
#     df[column_name + '_month'] = df[column_name].apply(get_month)
#     df[column_name + '_day'] = df[column_name].apply(get_day)

#     df[column_name + '_year'].fillna(0, inplace=True)
#     df[column_name + '_month'].fillna(0, inplace=True)
#     df[column_name + '_day'].fillna(0, inplace=True)

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

def process_labels_binary(df, column_name, prefix):
    new_column_name = prefix  + column_name
    df[new_column_name] = (df[column_name] != 'NDF').astype(int)

def process_labels_category(df, column_name, prefix):
    new_column_name = prefix  + column_name
    df[new_column_name] = df[column_name].apply(lambda x: LABEL_MAPPING[x])

def save_metadata(median_age, df_feature_columns):
    meta_data = {}
    meta_data['median_age'] = median_age
    meta_data['feature_columns'] = df_feature_columns

    with open(METADATA_PATH, 'wb') as file:
        pkl.dump(meta_data, file)


def load_metadata():
    with open(METADATA_PATH, 'rb') as file:
        meta_data = pkl.load(file)

    return meta_data


def convert_to_useful_attributes(df, column_name, attributes):
    df[column_name] = df[column_name].apply(lambda x : 'Other' if x not in attributes else x)
    return df


def session_features(df, df_ses):
    total_seconds = df_ses.groupby('user_id')['secs_elapsed'].sum()
    average_seconds = df_ses.groupby('user_id')['secs_elapsed'].mean().fillna(0)
    total_sessions = df_ses.groupby('user_id')['action'].count()
    distinct_sessions = df_ses.groupby('user_id')['action'].nunique()
    num_short_sessions = df_ses[df_ses['secs_elapsed'] <= 300].groupby('user_id')['action'].count()
    num_long_sessions = df_ses[df_ses['secs_elapsed'] >= 2000].groupby('user_id')['action'].count()
    num_devices = df_ses.groupby('user_id')['device_type'].nunique()

    df['total_seconds'] = df['id'].apply(lambda x: total_seconds[x] if x in total_seconds else 0)
    df['average_seconds'] = df['id'].apply(lambda x: average_seconds[x] if x in average_seconds else 0)
    df['total_sessions'] = df['id'].apply(lambda x: total_sessions[x] if x in total_sessions else 0)
    df['distinct_sessions'] = df['id'].apply(lambda x: distinct_sessions[x] if x in distinct_sessions else 0)
    df['num_short_sessions'] = df['id'].apply(lambda x: num_short_sessions[x] if x in num_short_sessions else 0)
    df['num_long_sessions'] = df['id'].apply(lambda x: num_long_sessions[x] if x in num_long_sessions else 0)
    df['num_devices'] = df['id'].apply(lambda x: num_devices[x] if x in num_devices else 0)
    return df


def preprocess_df(df, train=True):
    # check validity in test set?
    # extract_date_features(df, 'date_first_booking')

    # read these values from a file for test

    median_age = None
    df_feature_columns = None
    if not train:
        meta_data = load_metadata()
        median_age = meta_data['median_age']
        df_feature_columns = meta_data['feature_columns']

    process_labels_category(df, 'country_destination', 'label_')
    median_age = process_age(df, 'age', 'processed_', median_age=median_age)

    for col in STAT_COLS:
        df = convert_to_useful_attributes(df, col, LIST_MAPPING[col])
        to_dummy(df, col)

    if train:
        df = df.drop(columns=DROP_COLUMNS)
        df_feature_columns = list(df.columns.values)
        df_feature_columns.remove(LABEL_COLUMN)
        save_metadata(median_age, df_feature_columns)    
    
    x_features = df[df_feature_columns].values
    y_labels = df[[LABEL_COLUMN]].values

    return x_features, y_labels.flatten()