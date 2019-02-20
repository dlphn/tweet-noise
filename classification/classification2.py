# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 14:28 2018

@author: dshi, hbaud, vlefranc
"""

import pandas as pd
import datetime
import sys
sys.path.append('..')
from config import FILEDIR
import os
from sklearn.preprocessing import LabelEncoder, robust_scale

pd.set_option('display.width', None)


class Classification:

    def __init__(self):
        self.current_file = FILEDIR + "tweets_newtypes.csv"
        self.columns = ['nb_follower', 'nb_following', 'verified', 'reputation', 'age', 'nb_tweets', 'posted_at',
                        'proportion_spamwords', 'proportion_whitewords', 'orthographe', 'nb_hashtag', 'guillemets',
                        'nb_emoji','named_id', 'type' ,'spam']
        self.df_tweets = None
        self.df_tweets_categorized = None

    def create_dataframe(self, categorize=True):
        df = pd.read_csv(self.current_file, encoding="utf-8")
        df['nb_follower'] = self.normalisation(df['nb_follower'])
        df['nb_following'] = self.normalisation(df['nb_following'])
        df['age'] = self.normalisation(df['age'])
        df['nb_tweets'] = self.normalisation(df['nb_tweets'])
        df['nb_follower'] = self.normalisation(df['nb_follower'])
        self.df_tweets = df[self.columns]
        self.df_tweets_categorized = self.df_tweets.copy(deep=True)
        self.categorize_columns(['posted_at'], self.categorize_time)
        if categorize:
            self.categorize_columns(['verified'], self.categorize_bool)
            self.categorize_columns(['spam'], self.categorize_bool)
            self.categorize_columns(['type'], self.categorize_type)
        # print(self.df_tweets_categorized.head())
        # print(type(df_tweets_categorized))
        self.df_tweets_categorized.to_csv(FILEDIR+'newtypes_categorized.csv')
        return 'Finished'

    def normalisation(self,X):
        NormX = robust_scale(X)
        return NormX

    def categorize_bool(self, x):
        if x:
            return 1
        else:
            return 0

    def categorize_type(self, x):
        if x == "actualité" :
            return 1
        if x == "reaction" :
            return 2
        if x == "conversation" :
            return 5
        if x == "publicité" :
            return 6
        if x == "bot" :
            return 7
        if x == "other spam" :
            return 8

    def categorize_time(self, x):
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

    def categorize_columns(self, cols, func):
        for col in cols:
            self.df_tweets_categorized[col] = self.df_tweets[col].apply(func)
            # print(self.df_tweets[col])


if __name__ == "__main__":

    classification = Classification()
    classification.create_dataframe()
    #df = pd.read_csv('C:\\Users\\Public\\Documents\\tweets_newtypes.csv')
    #print(df.head())
    #print(df.groupby('type').id.count())



