from settings import *
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np


def split_data(data, test_size=0.4):
    new_train, new_test = train_test_split(data, test_size=test_size, shuffle=True, random_state=1)
    return new_train, new_test

def process_labels_binary(df, column_name):
    df[column_name] = (df[column_name] != LABEL_MAPPING['NDF']).astype(int)

class ClassifyLogistic():
    def __init__(self, train_df):
        process_labels_binary(train_df, LABEL_COLUMN)
        new_train_df, new_test_df = split_data(train_df, test_size=0.6)
        self.index = new_test_df.index

        self.xtrain, self.ytrain = self.get_features_labels(new_train_df)
        # print("Train bincount", np.bincount(self.ytrain))
        self.xtest, self.ytest = self.get_features_labels(new_test_df)
        # print("Test bincount", np.bincount(self.ytest))

        self.logistic = LogisticRegression(penalty='l1', solver='liblinear', random_state=1, max_iter=100)

    def get_new_train_index(self):
        return self.index

    def get_features_labels(self, df):
        return df.drop(columns=[LABEL_COLUMN]).values, df[[LABEL_COLUMN]].values.flatten()
    
    def train(self):
        self.logistic.fit(self.xtrain, self.ytrain)
        self.eval(self.xtest, self.ytest)

    def pred(self, test_df):
        process_labels_binary(test_df, LABEL_COLUMN)
        xtest, ytest = self.get_features_labels(test_df)
        accuracy_test, log_loss = self.eval(xtest, ytest)
        print("Predictions...")
        print("\t Accuracy {} Log_Loss {}".format(accuracy_test, log_loss))

        return self.logistic.predict(xtest)

    def eval(self, xtest, ytest):
        ypred = self.logistic.predict(xtest)
        # print(np.bincount(ypred))

        yprob = self.logistic.predict_proba(xtest)
        accuracy_test = metrics.accuracy_score(ytest, ypred)
        log_loss = metrics.log_loss(ytest, yprob)

        return accuracy_test, log_loss

