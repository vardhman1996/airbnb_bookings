import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from data_processing import *
from statistics import *

DATA_PATH = './data/'
METADATA_PATH = './metadata/'

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
    new_train, new_test = train_test_split(train_users, test_size=0.3, shuffle=True, random_state=0)
    return new_train, new_test


# countries = pd.read_csv(DATA_PATH + 'countries.csv')
# age_gender = pd.read_csv(DATA_PATH + 'age_gender_bkts.csv')

# print(train_users.columns.values)

train_users_df = load_data()
sessions_agg = load_sessions(agg=True)

merged_dataset_df = merge_df(train_users_df, sessions_agg, left_column='id', right_column='user_id', how='left')

train_df, test_df = split_data(merged_dataset_df)

# xtrain, ytrain = preprocess_df(train_df, train=True)
# xtest, ytest = preprocess_df(test_df, train=False)


calc_stats(train_df)

# print(ytrain[:100])
# clf = LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial').fit(xtrain, ytrain)

# ypred = clf.predict(xtest)

# print(np.bincount(ytest))
# print(np.bincount(ypred))

# acc_test = clf.score(xtest, ytest)
# print("test ", acc_test)

# ypred = clf.predict(xtrain)

# print(np.bincount(ytrain))
# print(np.bincount(ypred))

# acc_train = clf.score(xtrain, ytrain)
# print("train ", acc_train)