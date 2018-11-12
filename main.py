import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
# %matplotlib inline
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import multiprocessing 
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

def plot_confusion_matrix(cm, classes, filename,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def evaluate_model(ypred, ytest):
    print(np.bincount(ytest))
    print(np.bincount(ypred))

    accuracy_test = metrics.accuracy_score(ytest, ypred)
    confusion_matrix = metrics.confusion_matrix(ytest, ypred)
    f1_score = metrics.f1_score(ytest, ypred, average='weighted')

    print("Accuracy: {}".format(accuracy_test))

    labels = [None] * len(LABEL_MAPPING)
    for k, v in LABEL_MAPPING.items():
        labels[v] = k
    print("Confusion matrix")
    plot_confusion_matrix(confusion_matrix, labels, os.path.join(METRICS, "{}_cm.png".format(name)))

    print("f1 score {}".format(f1_score))

# countries = pd.read_csv(DATA_PATH + 'countries.csv')
# age_gender = pd.read_csv(DATA_PATH + 'age_gender_bkts.csv')

train_users_df = load_data()
# sessions_agg = load_sessions(agg=True)
session_df = load_sessions()

# merged_dataset_df = merge_df(train_users_df, sessions_agg, left_column='id', right_column='user_id', how='left')

train_df, test_df = split_data(train_users_df)

train_df = session_features(train_df, session_df)
test_df = session_features(test_df, session_df)

xtrain, ytrain = preprocess_df(train_df, train=True)
xtest, ytest = preprocess_df(test_df, train=False)

# parameters = {
#     'n_estimators': [100, 200],
#     'max_features': ['auto', 'log2'],
#     'max_depth': [3, 4, 5]
# }

logistic = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=1)
random_forest = RandomForestClassifier(n_estimators=50, random_state=1)
gaussian_nb = GaussianNB()
gradient_boosting = GradientBoostingClassifier()
votingclf = VotingClassifier(estimators=[('lr', logistic), ('rf', random_forest), ('gnb', gaussian_nb), ('gb', gradient_boosting)], voting='hard')

classifiers = [('Gaussian nb', gaussian_nb),('Random forest', random_forest), ('Gradient Boosting', gradient_boosting), ('Voting', votingclf)]

for name, classifier in classifiers:
    print("Running {} classifier...".format(name))
    classifier.fit(xtrain, ytrain)
    ypred = classifier.predict(xtest)
    evaluate_model(ypred, ytest)

# clf = GradientBoostingClassifier(n_estimators=1000, verbose=True).fit(xtrain, ytrain)


# grad_boost = GradientBoostingClassifier()

# clf = GridSearchCV(grad_boost, parameters, verbose=100, n_jobs=multiprocessing.cpu_count() - 1)
# clf.fit(xtrain, ytrain)

# print("Results {}".format(clf.cv_results_))
# print("Best params: {}".format(clf.best_params_))

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