import pandas as pd
import os
from sklearn.model_selection import train_test_split
from data_processing import *
from settings import *

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

def get_data():
    if not os.path.isfile(PROCESSED_DATA):
        train_users_df = load_data()
        session_df = load_sessions()
        train_df, test_df = split_data(train_users_df)
        train_df = session_features(train_df, session_df)
        test_df = session_features(test_df, session_df)

        xtrain, ytrain = preprocess_df(train_df, train=True)
        xtest, ytest = preprocess_df(test_df, train=False)

        data = {
            'xtrain': xtrain,
            'ytrain': ytrain,
            'xtest': xtest,
            'ytest': ytest,
        }
        print('Writing data to file')
        with open(PROCESSED_DATA, 'wb') as file:
            pkl.dump(data, file)
    else:
        print('Loading data from file')
        with open(PROCESSED_DATA, 'rb') as file:
            data = pkl.load(file)
        xtrain, ytrain, xtest, ytest = data['xtrain'], data['ytrain'], data['xtest'], data['ytest']
        

    return xtrain, ytrain, xtest, ytest