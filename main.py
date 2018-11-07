import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
import numpy as np

from data_processing import *

DATA_PATH = '~/.kaggle/'

# countries = pd.read_csv(DATA_PATH + 'countries.csv')
# age_gender = pd.read_csv(DATA_PATH + 'age_gender_bkts.csv')
train_users = pd.read_csv(DATA_PATH + 'train_users_2.csv')
# sessions = pd.read_csv(DATA_PATH + 'sessions.csv')
test_users = pd.read_csv(DATA_PATH + 'test_users.csv')

print(train_users.columns.values)

# preprocess_df(train_users)
# print(train_users.columns)