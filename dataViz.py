# -*- coding: utf-8 -*-
"""
Created on Thu Nov 1 18:06 2018

@author: dshi, hbaud, vlefranc
"""

import pandas as pd
import datetime
from config import FILEDIR
import numpy as np
import matplotlib.pyplot as plt


pd.set_option('display.width', None)

current_file = FILEDIR + "tweets_2018-10-31T14:25:32.735253.csv"
df = pd.read_csv(current_file, encoding="utf-8")
# print(df.head())
# print(df.describe())

columns = ['nb_follower', 'nb_following', 'verified', 'reputation', 'age', 'nb_tweets', 'time', 'proportion_spamwords',
       'orthographe', 'nb_emoji', 'RT', 'spam']
df_tweets = df[columns]
# print(df_tweets.dtypes)

def categorize_proportion(x):
    if x > 0.5:
        return 1
    else:
        return 0


def categorize_bool(x):
    if x:
        return 1
    else:
        return 0


def categorize_time(x):
    now = datetime.datetime.now()
    today8am = now.replace(hour=8, minute=0, second=0, microsecond=0)
    todaynoon = now.replace(hour=12, minute=0, second=0, microsecond=0)
    today2pm = now.replace(hour=14, minute=0, second=0, microsecond=0)
    today6pm = now.replace(hour=18, minute=0, second=0, microsecond=0)
    today10pm = now.replace(hour=22, minute=0, second=0, microsecond=0)
    t = x.split(':')
    time = now.replace(hour=int(t[0]), minute=int(t[1]), second=int(t[2]), microsecond=0)
    if today8am <= time < todaynoon:
        return 'matin'
    elif todaynoon <= time < today2pm:
        return 'midi'
    elif today2pm <= time < today6pm:
        return 'après-midi'
    elif today6pm <= time < today10pm:
        return 'soir'
    else:
        return 'nuit'


def categorize_columns(cols, func):
    for col in cols:
        df_tweets_categorized[col] = df_tweets[col].apply(func)


def split(feature, categorize):
    df_feature = df_tweets_categorized[[feature, 'spam']]
    df_spam = df_feature[df_tweets_categorized["spam"]]
    df_nospam = df_feature[df_tweets_categorized["spam"] == False]
    print(df_spam.head())
    print(df_nospam.head())
    counts_spam = []
    counts_nospam = []
    if categorize:
        categories = df_feature[feature].unique()

        for i in categories:
            counts_spam.append((i, df_spam[feature].value_counts()[i]))
            counts_nospam.append((i, df_nospam[feature].value_counts()[i]))

        df_stats_spam = pd.DataFrame(counts_spam, columns=[feature, 'number_of_tweets']).set_index(feature)
        df_stats_nospam = pd.DataFrame(counts_nospam, columns=[feature, 'number_of_tweets']).set_index(feature)
    else:
        df_stats_spam = df_spam
        df_stats_nospam = df_nospam
    DF = pd.merge(df_stats_spam, df_stats_nospam, on=[feature])
    print(DF)
    DF.plot.bar()

    plt.show()


df_tweets_categorized = df_tweets.copy(deep=True)
# categorize_columns(['reputation', 'proportion_spamwords', 'orthographe'], categorize_proportion)
# categorize_columns(['verified', 'RT', 'spam'], categorize_bool)
categorize_columns(['time'], categorize_time)
# print(df_tweets_categorized.head())

# df2 = pd.DataFrame(df_tweets_categorized, columns=['time', 'spam'])
# df2.plot.bar()

# split('nb_follower', False)

cond = df_tweets_categorized['spam'] == True
subset_a = df_tweets_categorized[cond].dropna()
subset_b = df_tweets_categorized[~cond].dropna()
# plt.scatter(subset_a['nb_following'], subset_a['nb_follower'], s=60, c='b', label='Spam')
# plt.scatter(subset_b['nb_following'], subset_b['nb_follower'], s=60, c='r', label='Not spam')
plt.legend()

plt.show()
