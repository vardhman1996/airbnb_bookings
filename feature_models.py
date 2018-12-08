from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

from settings import *


# linear =
# less than 1 (0,)
# greater than 115 (0,)
# min and max 18.409835818231183 53.424355055147494
# difference greater than 5 (58087,)
# median 37.14333985791983

# Huber =
# less than 1 (7,)
# greater than 115 (1865,)
# min and max -35.94206115621723 1947.131627741896
# difference greater than 5 (59137,)
# median 28.060843670406097

# ElasticNet =
# less than 1 (0,)
# greater than 115 (0,)
# min and max 8.27978563285912 45.48441713489797
# difference greater than 5 (60667,)
# median 37.33353090440245

# Lasso =
# less than 1 (0,)
# greater than 115 (0,)
# min and max 9.72976672044365 43.98109797547863
# difference greater than 5 (61412,)
# median 37.227065988288544


def split_data(data, test_size=0.4):
    new_train, new_test = train_test_split(data, test_size=test_size, shuffle=True, random_state=1)
    return new_train, new_test

def process_labels_binary(df, column_name):
    df[column_name] = (df[column_name] != LABEL_MAPPING['NDF']).astype(int)


class PredictAge:
    def __init__(self, train_df):
        self.index = train_df.index
        self.naindex = train_df[train_df[AGE_COLUMN].isnull()].index
        xtrain = train_df.dropna()
        print(min(xtrain[AGE_COLUMN].values))
        ytrain = xtrain[AGE_COLUMN]
        xtrain = xtrain.drop(columns=[AGE_COLUMN])
        self.xtrain = self.get_features_labels(xtrain)
        self.ytrain = self.get_features_labels(ytrain)
        self.svr = SVR(kernel='rbf', gamma='scale')

    def get_train_index(self):
        return self.index

    def get_na_index(self):
        return self.naindex

    def get_features_labels(self, df):
        return df.values

    def train(self):
        self.svr.fit(self.xtrain, self.ytrain)
        self.eval(self.xtrain, self.ytrain, evaluate=True)

    def eval(self, x, y_true, evaluate):
        y_pred = self.svr.predict(x)
        if evaluate:
            print(metrics.mean_squared_error(y_true, y_pred))
        else:
            return y_pred

    def pred(self, test_df, evaluate=False):
        y_test = test_df[AGE_COLUMN]
        x_test = test_df.drop(columns=[AGE_COLUMN])
        x_test = self.get_features_labels(x_test)
        y_test = self.get_features_labels(y_test)
        out = self.eval(x_test, y_test, evaluate)
        return out


class ClassifyLogistic:
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
        yprob = self.logistic.predict_proba(xtest)
        accuracy_test = metrics.accuracy_score(ytest, ypred)
        log_loss = metrics.log_loss(ytest, yprob)

        return accuracy_test, log_loss

