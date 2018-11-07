import pandas as pd
import numpy as np

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

def extract_date_features(df, column_name):
    df[column_name + '_year'] = df[column_name].apply(get_year)
    df[column_name + '_month'] = df[column_name].apply(get_month)
    df[column_name + '_day'] = df[column_name].apply(get_day)

    df[column_name + '_year'].fillna(0, inplace=True)
    df[column_name + '_month'].fillna(0, inplace=True)
    df[column_name + '_day'].fillna(0, inplace=True)

def process_age(df, column_name, prefix):
    new_column_name = prefix + column_name
    df[new_column_name] = df[column_name]
    column_data = np.array(df[new_column_name].values.tolist())
    df[new_column_name] = np.where(column_data > 115, np.NaN, column_data).tolist()
    df_median = df[new_column_name].median()
    print("median: {}".format(df_median))

    df[new_column_name].fillna(df_median, inplace=True)
    assert(df[new_column_name].isna().sum() == 0)

def preprocess_df(df):
    # check validity in test set?
    # extract_date_features(df, 'date_first_booking')

    process_age(df, 'age', 'processed_')
    to_dummy(df, 'gender', 'gender')
    to_categorical(df, 'first_browser', 'processed_')
    to_dummy(df, 'first_device_type', 'first_device_type')
    to_dummy(df, 'signup_app', 'signup_type')
    to_dummy(df, 'signup_method', 'signup_method')
    to_categorical(df, 'signup_flow', 'processed_')
    to_dummy(df, 'language', 'language_used')
    to_dummy(df, 'affiliate_channel', 'affiliate_channel')
    to_dummy(df, 'signup_app', 'signup_app')
    to_dummy(df, 'first_affiliate_tracked', 'first_aff_tracked')