import os

from sklearn.model_selection import train_test_split

import feature_models as feat_model
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
    new_train, new_test = train_test_split(train_users, test_size=0.3, shuffle=True, random_state=1)
    return new_train, new_test


def get_data():
    if not os.path.isfile(PROCESSED_DATA):
        train_users_df = load_data()
        session_df = load_sessions()
        train_df, test_df = split_data(train_users_df)

        train_df, sess_metadata = session_features(train_df, session_df, metadata=None)
        test_df, _ = session_features(test_df, session_df, metadata=sess_metadata)

        xtrain_df, ytrain_df = preprocess_df(train_df, train=True)
        xtest_df, ytest_df = preprocess_df(test_df, train=False)

        predict_age = feat_model.PredictAge(xtrain_df)
        predict_age.train()

        na_train_index = predict_age.get_na_index()

        xtrain_age_feature = predict_age.pred(xtrain_df.loc[na_train_index])

        for idx, val in zip(na_train_index, xtrain_age_feature):
            tmp = val
            if val < 1:
                tmp = 1
            elif val > 115:
                tmp = 115
            xtrain_df.at[idx, AGE_COLUMN] = int(round(tmp))

        na_test_index = test_df[test_df[AGE_COLUMN].isnull()].index
        xtest_age_feature = predict_age.pred(xtest_df)
        
        for idx, val in zip(na_test_index, xtest_age_feature):
            tmp = val
            if val < 1:
                tmp = 1
            elif val > 115:
                tmp = 115
            xtest_df.at[idx, AGE_COLUMN] = int(round(tmp))

        # APPLY LOGISTIC ON XTRAIN_DF AND YTRAIN_DF TO GET MORE FEATURES
        classify_logistic = feat_model.ClassifyLogistic(pd.concat([xtrain_df, ytrain_df], axis=1))
        classify_logistic.train()

        new_train_index = classify_logistic.get_new_train_index()
        xtrain_df, ytrain_df = xtrain_df.loc[new_train_index], ytrain_df.loc[new_train_index]

        xtrain_logit_feature = classify_logistic.pred(pd.concat([xtrain_df, ytrain_df], axis=1))
        xtrain_df['logistic_feature'] = xtrain_logit_feature

        xtest_logit_feature = classify_logistic.pred(pd.concat([xtest_df, ytest_df], axis=1))
        xtest_df['logistic_feature'] = xtest_logit_feature

        xtrain = xtrain_df.values
        ytrain = ytrain_df.values.flatten()
        xtest = xtest_df.values
        ytest = ytest_df.values.flatten()

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