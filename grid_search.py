import numpy as np
from sklearn.model_selection import RandomizedSearchCV

def random_search(classifiers, params_dict, x, y):
    n_iter_search = 3
    for name, clf in classifiers:
        print("Classifier: {}".format(name))
        param_dict = params_dict[name]
        random_search = RandomizedSearchCV(clf, param_distributions=param_dict,
                                           n_iter=n_iter_search, cv=2, scoring='fowlkes_mallows_score', verbose=1, n_jobs=-1)
        random_search.fit(x, y)
        report(random_search.cv_results_, n_top=5)


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
