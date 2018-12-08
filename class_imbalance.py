from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler


def get_simple_data(xtrain, ytrain):
    return xtrain, ytrain


def get_over_sampled_data(xtrain, ytrain):
    ros = RandomOverSampler(random_state=0)
    x_train_resampled, y_train_resampled = ros.fit_resample(xtrain, ytrain)
    return x_train_resampled, y_train_resampled


def get_smoteenn_data(xtrain, ytrain):
    sme = SMOTEENN(random_state=0)
    x_train_resampled, y_train_resampled = sme.fit_resample(xtrain, ytrain)
    return x_train_resampled, y_train_resampled


def get_smote_data(xtrain, ytrain):
    sm = SMOTE(random_state=0)
    x_train_resampled, y_train_resampled = sm.fit_resample(xtrain, ytrain)
    return x_train_resampled, y_train_resampled


def get_under_sampled_data(xtrain, ytrain):
    rus = RandomUnderSampler(random_state=0)
    x_train_resampled, y_train_resampled = rus.fit_resample(xtrain, ytrain)
    return x_train_resampled, y_train_resampled