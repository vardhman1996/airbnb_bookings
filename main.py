import pandas as pd
from datetime import date
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost.sklearn import XGBClassifier
from sklearn.svm import SVC
from sklearn import metrics
import multiprocessing 
from statistics import *
from class_imbalance import *
from data_loader import get_data
from plot_graphs import *
from settings import SAVE_METRICS
from imblearn.metrics import classification_report_imbalanced as imbal_class_report
from scipy.stats import randint as sp_randint
import grid_search

def evaluate_model(ypred, yprob, ytest, filename):
    print(np.bincount(ytest))
    print(np.bincount(ypred))

    labels = [None] * len(LABEL_MAPPING)
    for k, v in LABEL_MAPPING.items():
        labels[v] = k

    accuracy_test = metrics.accuracy_score(ytest, ypred)
    confusion_matrix = metrics.confusion_matrix(ytest, ypred)
    log_loss = metrics.log_loss(ytest, yprob)
    f1_score = metrics.f1_score(ytest, ypred, average='weighted')
    classification_report = metrics.classification_report(ytest, ypred, target_names=labels)
    imbalance_classification_report = imbal_class_report(ytest, ypred, target_names=labels)

    print("Accuracy: {}".format(accuracy_test))
    print("f1 score {}".format(f1_score))
    print("Log Loss {}".format(log_loss))

    if SAVE_METRICS:
        print("Confusion matrix")
        plot_confusion_matrix(confusion_matrix, labels, os.path.join(METRICS, "{}_cm.png".format(filename)))
        with open(os.path.join(CLASSIFICATION_REPORT, "{}.txt".format(filename)), 'w') as txtfile:
            txtfile.write(classification_report)
        with open(os.path.join(IMBALANCE_CLASSIFICATION_REPORT, "{}.txt".format(filename)), 'w') as txtfile:
            txtfile.write(imbalance_classification_report)

    print(classification_report)
    print(imbalance_classification_report)

def train_models(classifiers, xtrain, ytrain, xtest, ytest, transform=None, sample_weight=None):
    for name, classifier in classifiers:
        print("Running {} classifier...".format(name))
        classifier.fit(xtrain, ytrain, sample_weight=sample_weight)
        ypred = classifier.predict(xtest)
        yprob = classifier.predict_proba(xtest)
        filename = '{}/{}'.format(transform, name)
        evaluate_model(ypred, yprob, ytest, filename)

def get_classifiers():
    logistic = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=1, max_iter=1000, verbose=1)
    random_forest = RandomForestClassifier(n_estimators=100, random_state=1, verbose=1, n_jobs=-1)
    gaussian_nb = GaussianNB()
    gradient_boosting = GradientBoostingClassifier(max_features='auto', verbose=True)
    # svc = SVC(kernel='rbf')
    votingclf = VotingClassifier(estimators=[('lr', logistic), ('rf', random_forest), ('gnb', gaussian_nb), ('gb', gradient_boosting)], voting='soft', n_jobs=-1)
    mlp = MLPClassifier((256, 512, 1024, 512, 256), max_iter=100, learning_rate='adaptive', verbose=True, validation_fraction=0.1)
    bagging_classifier = BaggingClassifier(n_jobs=-1, verbose=1)
    adaboost = AdaBoostClassifier(random_state=1)
    extraTree = ExtraTreesClassifier(n_jobs=-1, class_weight='balanced', verbose=1)
    xgboost = XGBClassifier(n_jobs=-1, verbose=1)

    classifiers = [ 
                    ('logistic', logistic),
                    # ('random_forest', random_forest),
                    # ('adaboost', adaboost),
                    # ('xgboost', xgboost)
                ]

    params_dict = {
        # 'gradient_boosting': {
        #     'n_estimators' : sp_randint(50, 1000),
        #     'max_depth': sp_randint(1, 11),
        #     'max_features': ['auto', 'sqrt', 'log2', None]
        # },
        'adaboost': {
            'n_estimators': sp_randint(50, 1000)
        },
        'xgboost': {
            'n_estimators': sp_randint(50, 1000),
            'max_depth': sp_randint(1, 11),
            'booster': ['gbtree', 'gblinear', 'dart']
        }
    }
    return classifiers, params_dict

if __name__ == "__main__":
    xtrain, ytrain, xtest, ytest = get_data()
    print("Using sampling technique: {}".format(SAMPLING_METHOD))
    xtrain_resampled, ytrain_resampled = SAMPLING_MAPPING[SAMPLING_METHOD](xtrain, ytrain)
    if RANDOM_SEARCH:
        classifiers, params_dict = get_classifiers()
        grid_search.random_search(classifiers, params_dict, xtrain_resampled, ytrain_resampled)
    else:
        classifiers, _ = get_classifiers()

    if not os.path.exists(os.path.join(METRICS, SAMPLING_METHOD)):
        os.makedirs(os.path.join(METRICS, SAMPLING_METHOD))
    if not os.path.exists(os.path.join(CLASSIFICATION_REPORT, SAMPLING_METHOD)):
        os.makedirs(os.path.join(CLASSIFICATION_REPORT, SAMPLING_METHOD))
    if not os.path.exists(os.path.join(IMBALANCE_CLASSIFICATION_REPORT, SAMPLING_METHOD)):
        os.makedirs(os.path.join(IMBALANCE_CLASSIFICATION_REPORT, SAMPLING_METHOD))

    print("Original bincount")
    print(np.bincount(ytrain))
    print("Resampled bincount")
    print(np.bincount(ytrain_resampled))

    sample_weight = None
    if SAMPLING_METHOD == 'no_transform' and WEIGHTED:
        bincounts = np.bincount(ytrain)
        fit_weights = 1.0 / (bincounts / np.sum(bincounts))
        sample_weight = np.array([fit_weights[val] for val in ytrain_resampled])

    train_models(classifiers, xtrain_resampled, ytrain_resampled, xtest, ytest, transform=SAMPLING_METHOD, sample_weight=sample_weight)
