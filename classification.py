# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 14:28 2018

@author: dshi, hbaud, vlefranc
"""

import pandas as pd
import datetime
from config import FILEDIR

from sklearn.preprocessing import LabelEncoder

pd.set_option('display.width', None)

current_file = FILEDIR + "tweets_2018-10-31T14:25:32.735253.csv";
df = pd.read_csv(current_file, encoding="utf-8")
# print(df.head())
# print(df.describe())

columns = ['nb_follower', 'nb_following', 'verified', 'reputation', 'age', 'nb_tweets', 'time', 'proportion_spamwords',
       'orthographe', 'nb_emoji', 'RT', 'spam']
df_tweets = df[columns]


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


df_tweets_categorized = df_tweets.copy(deep=True)
categorize_columns(['reputation', 'proportion_spamwords', 'orthographe'], categorize_proportion)
categorize_columns(['verified', 'RT', 'spam'], categorize_bool)
categorize_columns(['time'], categorize_time)
print(df_tweets_categorized.head())

for col in df_tweets_categorized.columns.values:
    print(col, df_tweets_categorized[col].unique())


def label_encode(data_frame, cols):
    for col in cols:
        le = LabelEncoder()
        col_values_unique = list(data_frame[col].unique())
        le_fitted = le.fit(col_values_unique)

        col_values = list(data_frame[col].values)
        le.classes_
        col_values_transformed = le.transform(col_values)
        data_frame[col] = col_values_transformed


# df_tweets_ohe = df_tweets.copy(deep=True)
# to_be_encoded_cols = df_tweets_ohe.columns.values
# label_encode(df_tweets_ohe, to_be_encoded_cols)
# print(df_tweets_ohe.head())
