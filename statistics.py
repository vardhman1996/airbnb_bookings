import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib as ml
ml.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.feature_selection import chi2
plt.style.use('seaborn')
import seaborn as sns
from settings import *
import os


def process_labels_category(y):
    y[LABEL_COLUMN] = (y['country_destination'] != 'NDF').astype(int)

def calc_booking_stats(df_train):
    df_train['booked'] = (df_train['country_destination'] != 'NDF').astype(int)
    df_train['non_booked'] = (df_train['country_destination'] == 'NDF').astype(int)


    for col in STAT_COLS:
        df_book_stats = df_train[['id', col, 'booked', 'non_booked']]
        df_group = df_book_stats.groupby(col)['booked', 'non_booked'].sum()
        df_group = df_group.div(df_group.sum(1), axis=0)
        ax = df_group.plot.bar(stacked=True, figsize=(12,10))
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + 0.1, box.width * 0.8, box.height * 0.9])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


        plt.xlabel('Categories: {}'.format(col), fontsize=16)
        plt.ylabel('Relative counts', fontsize=16)
        plt.title('{} vs relative counts'.format(col), fontsize=16)
        plt.savefig(os.path.join(STACK_PLOTS, '{}.png'.format(col)))
        print(df_group.head())
        # break


def calc_stats(df_train):
    # y = df_train[[LABEL_COLUMN]]
    # process_labels_category(y)

    x = df_train[STAT_COLS + ['country_destination']]
    df_inf = df_train[(df_train['country_destination'] != 'NDF')]
    p_vals = []
    for col in STAT_COLS:
        df_stat_check = df_inf[['id', col, 'country_destination']]

        df_val_counts = df_inf[col].value_counts()
        plt.figure(figsize=(20,18))
        plot = sns.barplot(x=df_val_counts.index, y=df_val_counts)
        plot.tick_params(labelsize=16)
        plt.xticks(rotation=90)
        plt.xlabel('Categories: {}'.format(col), fontsize=16)
        plt.ylabel('Value counts', fontsize=16)
        plt.title('{} vs value counts'.format(col), fontsize=16)
        plt.savefig(os.path.join(GRAPHS, '{}.png'.format(col)))

        observed = df_stat_check.pivot_table('id', [col], 'country_destination', aggfunc='count').reset_index()
        observed = observed.set_index(col)
        observed = observed.fillna(0)

        chi2, p, dof, expected = stats.chi2_contingency(observed)

        p_vals += [(col, p)]
    print("Pvalues {}".format(p_vals))
