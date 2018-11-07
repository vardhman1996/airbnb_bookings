import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
import numpy as np
from sklearn.model_selection import train_test_split

from data_processing import *

DATA_PATH = '~/.kaggle/'

def split_data(train_users):
    train_users = train_users.sort_values(by='date_account_created')
    new_train, new_test = train_test_split(train_users, test_size=0.3, shuffle=False)
    return new_train, new_test


# countries = pd.read_csv(DATA_PATH + 'countries.csv')
# age_gender = pd.read_csv(DATA_PATH + 'age_gender_bkts.csv')
train_users = pd.read_csv(DATA_PATH + 'train_users_2.csv')
# sessions = pd.read_csv(DATA_PATH + 'sessions.csv')
test_users = pd.read_csv(DATA_PATH + 'test_users.csv')

print(train_users.columns.values)

# preprocess_df(train_users)
# print(train_users.columns)