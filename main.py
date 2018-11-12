import pandas as pd
from datetime import date
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
import multiprocessing 
from statistics import *
from class_imbalance import *
from data_loader import get_data
from plot_graphs import *

def evaluate_model(ypred, ytest, name):
    print(np.bincount(ytest))
    print(np.bincount(ypred))

    labels = [None] * len(LABEL_MAPPING)
    for k, v in LABEL_MAPPING.items():
        labels[v] = k

    accuracy_test = metrics.accuracy_score(ytest, ypred)
    confusion_matrix = metrics.confusion_matrix(ytest, ypred)
    f1_score = metrics.f1_score(ytest, ypred, average='weighted')
    classification_report = metrics.classification_report(ytest, ypred, target_names=labels)
    print("Accuracy: {}".format(accuracy_test))
    print("Confusion matrix")
    plot_confusion_matrix(confusion_matrix, labels, os.path.join(METRICS, "{}_cm.png".format(name)))
    print("f1 score {}".format(f1_score))
    print(classification_report)

def train_models(classifiers, xtrain, ytrain, xtest, ytest, transform=None):
    print("Original bincount")
    print(np.bincount(ytrain))
    print("Resampled bincount")
    print(np.bincount(ytrain_resampled))

    for name, classifier in classifiers:
        print("Running {} classifier...".format(name))
        classifier.fit(xtrain_resampled, ytrain_resampled)
        ypred = classifier.predict(xtest)

        filename = '{}_{}'.format(name, transform)
        evaluate_model(ypred, ytest, filename)

def get_classifiers():
    logistic = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=1, max_iter=1000, verbose=1)
    random_forest = RandomForestClassifier(n_estimators=100, random_state=1, verbose=1, n_jobs=-1)
    gaussian_nb = GaussianNB()
    gradient_boosting = GradientBoostingClassifier(max_features='auto', verbose=True)
    svc = SVC(kernel='rbf')
    votingclf = VotingClassifier(estimators=[('lr', logistic), ('rf', random_forest), ('gnb', gaussian_nb), ('gb', gradient_boosting)], voting='hard', n_jobs=-1)
    
    # print("size: {}".format(size))
    mlp = MLPClassifier((500, 250), max_iter=10, learning_rate='adaptive', verbose=True, validation_fraction=0.1)

    # RUNNING ON ATTU
    # classifiers = [('gradient_boosting', gradient_boosting), ('Gaussian nb', gaussian_nb), ('Random forest', random_forest), ('Voting', votingclf)]

    classifiers = [('MLP', mlp)]
    return classifiers

if __name__ == "__main__":
    xtrain, ytrain, xtest, ytest = get_data()

    print("Using sampling technique: {}".format(SAMPLING_METHOD))
    xtrain_resampled, ytrain_resampled = SAMPLING_MAPPING[SAMPLING_METHOD](xtrain, ytrain)
    classifiers = get_classifiers()

    train_models(classifiers, xtrain_resampled, ytrain_resampled, xtest, ytest, transform=SAMPLING_METHOD)