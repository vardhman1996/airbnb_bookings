import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from settings import *
import os


def random_search(classifiers, params_dict, x, y, sample_weight=None):
    n_iter_search = 5
    for name, clf in classifiers:
        print("Classifier: {}".format(name))
        param_dict = params_dict[name]
        random_search = RandomizedSearchCV(clf, param_distributions=param_dict,
                                           n_iter=n_iter_search, cv=2, scoring='fowlkes_mallows_score', verbose=1, n_jobs=-1)
        random_search.fit(x, y, sample_weight=sample_weight)
        report(random_search.cv_results_, name, n_top=5)


def report(results, name, n_top=3):
    out = []
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)

        for candidate in candidates:
            out += ["Model with rank: {0}".format(i)]
            out += ["Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate])]
            out += ["Parameters: {0}".format(results['params'][candidate])]
            out += [""]

    out = '\n'.join(out)

    dirpath = os.path.join(os.path.join(GRID_SEARCH_REPORT, SAMPLING_METHOD), name)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    filename = "{}/result.txt".format(dirpath)
    with open(filename, 'w') as txtfile:
        txtfile.write(out)
    return None
