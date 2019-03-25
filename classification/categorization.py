# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 14:28 2018
@author: dshi, hbaud, vlefranc

Categorize features (normalisation, boolean, etc.).
"""

from sklearn.preprocessing import LabelEncoder, robust_scale
import pandas as pd
import datetime
import sys
from config import current_file, FILEDIR

sys.path.append('..')


pd.set_option('display.width', None)


def normalisation(X):
    norm_x = robust_scale(X)
    return norm_x


def categorize_bool(x):
    if x:
        return 0
    else:
        return 1


def categorize_type(tweet_type):
    types = ['actualité', 'reaction', 'conversation', 'other spam', 'publicité', 'bot']
    return types.index(tweet_type)


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
        return 0
    elif todaynoon <= time < today2pm:
        return 1
    elif today2pm <= time < today6pm:
        return 2
    elif today6pm <= time < today10pm:
        return 3
    else:
        return 4


class Classification:

    def __init__(self, labels='spam'):
        self.columns = ['id','nb_follower', 'nb_following', 'verified', 'reputation', 'age', 'nb_tweets', 'posted_at',
                        'length','proportion_spamwords', 'orthographe', 'nb_hashtag','nb_urls',
                        'nb_emoji','spam']
        self.df_tweets = None
        self.df_tweets_categorized = None
        self.labels = labels
        self.columns.append(labels)

    def create_dataframe(self, categorize=True):
        dataframe = pd.read_csv(current_file, encoding="utf-8")

        dataframe['nb_follower'] = normalisation(dataframe['nb_follower'])
        dataframe['nb_following'] = normalisation(dataframe['nb_following'])
        dataframe['age'] = normalisation(dataframe['age'])
        dataframe['nb_tweets'] = normalisation(dataframe['nb_tweets'])

        self.df_tweets = dataframe[self.columns]
        self.df_tweets_categorized = self.df_tweets.copy(deep=True)
        self.categorize_columns(['posted_at'], categorize_time)

        if categorize:
            # self.categorize_columns(['nb_emoji'], self.categorize_emoji)
            # if self.labels == 'spam':
            self.categorize_columns(['spam'], categorize_bool)
            if self.labels == 'type':
                self.categorize_columns(['type'], categorize_type)
            # self.categorize_columns(['nb_follower', 'nb_following'], self.categorize_follower_following)
            # self.categorize_columns(['age'], self.categorize_age)
            # self.categorize_columns(['nb_tweets'], self.categorize_nb_tweets)

        # print(self.df_tweets_categorized.head())
        # print(type(df_tweets_categorized))
        self.df_tweets_categorized.to_csv(FILEDIR + 'tweets_data2_categorized_spam.csv')
        return self.df_tweets_categorized

    def categorize_columns(self, cols, func):
        for col in cols:
            self.df_tweets_categorized[col] = self.df_tweets[col].apply(func)
            # print(self.df_tweets[col])


if __name__ == "__main__":

    classification = Classification('type')
    df = classification.create_dataframe()
    print(df.head())
