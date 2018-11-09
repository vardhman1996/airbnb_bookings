import pandas as pd
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.feature_selection import chi2
plt.style.use('seaborn')
import seaborn as sns
from settings import *
import os


def process_labels_category(y):
    y[LABEL_COLUMN] = y[LABEL_COLUMN].apply(lambda x: LABEL_MAPPING[x])




def calc_stats(df_train):
    # y = df_train[[LABEL_COLUMN]]
    # process_labels_category(y)


    x = df_train[STAT_COLS + ['country_destination']]
    df_inf = df_train[(df_train['country_destination'] != 'NDF') & (df_train['country_destination'] != 'other')]
    
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
