import matplotlib as ml
import itertools

ml.use("Agg")
import matplotlib.pyplot as plt

plt.style.use('ggplot')
import numpy as np
from settings import *
import os
import pandas as pd
from textwrap import wrap


def get_cmap(n, name='tab20'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


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
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_sampling_graphs():
    for sampling_types in os.listdir(IMBALANCE_CLASSIFICATION_REPORT):
        sampling_path = os.path.join(IMBALANCE_CLASSIFICATION_REPORT, sampling_types)

        geometric_means = []
        clfs = []
        class_gms = []
        supports = None

        files = sorted(os.listdir(sampling_path))
        for classification_report in files:
            if not classification_report.endswith('.txt'):
                continue

            classification_report_path = os.path.join(sampling_path, classification_report)
            report, last_line = classifaction_report_csv(open(classification_report_path, 'r').read())

            classifier_name = classification_report.split('.')[0]
            gm = last_line[5]

            class_gm_mean = report[:, 2].astype('float64')
            if supports is None:
                supports = report[:, -1]

            clfs += [classifier_name]
            geometric_means += [gm]
            class_gms += [class_gm_mean]

        class_gms = np.array(class_gms)
        # a = (class_gms * supports)
        # b = np.dot(class_gms, supports)[:, np.newaxis]
        # c = np.array(geometric_means)[:, np.newaxis]
        #
        # class_weights = (a * c) / b

        plot_bar_graph(clfs, class_gms, sampling_path)


def plot_final_graphs():
    for sampling_types in os.listdir(IMBALANCE_CLASSIFICATION_REPORT):
        sampling_path = os.path.join(IMBALANCE_CLASSIFICATION_REPORT, sampling_types)

        if sampling_types != 'no_transfrom_weighted' or sampling_path != 'over_sample':
            continue

        geometric_means = []
        clfs = []
        class_gms = []
        supports = None

        files = []
        for f in os.listdir(sampling_path):
            if f.endswith("final.txt"):
                files += [f]

        print(files)
        for classification_report in files:
            if not classification_report.endswith('.txt'):
                continue

            classification_report_path = os.path.join(sampling_path, classification_report)
            report, last_line = classifaction_report_csv(open(classification_report_path, 'r').read())

        #     classifier_name = classification_report.split('.')[0]
        #     gm = last_line[5]
        #
        #     class_gm_mean = report[:, 2].astype('float64')
        #     if supports is None:
        #         supports = report[:, -1]
        #
        #     clfs += [classifier_name]
        #     geometric_means += [gm]
        #     class_gms += [class_gm_mean]
        #
        # class_gms = np.array(class_gms)
        # a = (class_gms * supports)
        # b = np.dot(class_gms, supports)[:, np.newaxis]
        # c = np.array(geometric_means)[:, np.newaxis]
        #
        # class_weights = (a * c) / b

        plot_bar_graph(clfs, class_gms, sampling_path)


def plot_bar_graph(clfs, class_weights, dirpath):
    cumsum = class_weights.cumsum(axis=1)
    ind = np.arange(4)

    cmap = get_cmap(class_weights.shape[1])
    inverse_mapping = {}
    for k, v in LABEL_MAPPING.items():
        inverse_mapping[v] = k

    plt.bar(ind, class_weights[:, 0], 0.35, label=inverse_mapping[0], color=cmap(0))
    for i in range(1, class_weights.shape[1]):
        plt.bar(ind, class_weights[:, i], 0.35, bottom=i, label=inverse_mapping[i], color=cmap(i))

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0 + 0.05, box.y0, box.width * 0.8, box.height])

    plt.xlabel('Classifiers')
    plt.ylabel('Relative Geometric Mean')
    plt.title('\n'.join(wrap('G-mean for each class per classifier using {}'.format(os.path.basename(dirpath)), 60)))
    plt.xticks(ind, clfs)
    plt.ylim(0, 0.6)
    filename = os.path.join(dirpath, '{}_{}.png'.format(os.path.basename(dirpath), 'stacked'))
    plt.savefig(filename)
    plt.close()


def classifaction_report_csv(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split()
        row['class'] = row_data[0]
        row['pre'] = float(row_data[1])
        row['rec'] = float(row_data[2])
        row['spe'] = float(row_data[3])
        row['f1'] = float(row_data[4])
        row['geo'] = float(row_data[5])
        row['iba'] = float(row_data[6])
        row['sup'] = float(row_data[7])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    last_line = [float(val) for val in lines[-2].split()[3:-1]]
    last_line = ['avg'] + last_line

    return dataframe.values, last_line


if __name__ == '__main__':
    plot_sampling_graphs()
    plot_final_graphs()
