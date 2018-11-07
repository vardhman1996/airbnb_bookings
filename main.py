import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
import numpy as np
from sklearn.model_selection import train_test_split

from data_processing import *

DATA_PATH = './data/'


def load_data():
	train_users = pd.read_csv(DATA_PATH + 'train_users_2.csv')

	return train_users

def load_sessions(agg=False):
	sessions_df = pd.read_csv(DATA_PATH + 'sessions.csv')
	if agg:
		sessions_agg = agg_sessions(sessions_df)
		return sessions_agg

	return sessions_df


def split_data(train_users):
    train_users = train_users.sort_values(by='date_account_created')
    new_train, new_test = train_test_split(train_users, test_size=0.3, shuffle=False)
    return new_train, new_test


# countries = pd.read_csv(DATA_PATH + 'countries.csv')
# age_gender = pd.read_csv(DATA_PATH + 'age_gender_bkts.csv')

# print(train_users.columns.values)

train_users_df = load_data()
sessions_agg = load_sessions(agg=True)

# print(train_users_df.info())
# print(sessions_agg.info())
merged_result = merge_df(train_users_df, sessions_agg, left_column='id', right_column='user_id', how='left')
process_session_sec(merged_result, 'secs_elapsed', 'processed_')