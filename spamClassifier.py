# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 16:35 2019

@author: dshi, hbaud, vlefranc
"""

import pandas as pd
from pymongo import MongoClient
from features.Medias import MEDIAS, listemedias
from classification.random_forest import predict_rf
import sys


sys.path.append('..')
from config import FILEDIR, FILEBREAK, MONGODB

pd.set_option('max_columns', 10)


class Classification:

    #On initialise chaque tweet avec le dataframe et le modele entraine
    def __init__(self, dataframe, tweet, trained_model):
        self.tweet = tweet
        self.trained_model = trained_model
        self.df = dataframe


    #On regarde les parametres du cluster pour voir si on peut classer directement le tweet comme spam
    def classify_bycluster(self):
        num_cluster = self.tweet['pred'][0]
        print(num_cluster)
        if num_cluster == -1:
            self.set_cluster_spam(True)
        else :
            df_pred = self.df[self.df['pred']== num_cluster]
            length = df_pred.shape[0]
            if length < 4 :
                self.set_cluster_spam(False)
            else :
                df_pred.apply(self.categorize_label)
                if df.label.sum(axis = 0)/length > 0.1 :
                    self.set_cluster_spam(False)
                else :
                    if self.contains_media(df_pred) :
                        self.set_cluster_spam(False)
                    else :
                        self.set_cluster_spam(True)


    def set_cluster_spam(self, bool):
        if bool:
            self.tweet += 'spam,undefined'
        else:
            self.tweet += 'undefined,'
            self.tweet += str(self.classify_byrandomforest())

    def classify_byrandomforest(self):
        return predict_rf(self.trained_model, self.tweet)


    def categorize_label(self, x):
        if x == 'spam':
            return 0
        else:
            return 1

    def contains_media(self, df_pred):
        for index, row in df_pred.iterrows():
            if row['screen_name'] in listemedias :
                return True
        return False



df = pd.read_csv(FILEDIR+'clustering_2019-03-06_0.7_100000_100_base_fr_2.csv')
tweet = pd.read_csv(FILEDIR+'tweet.csv')


classif = Classification(df,tweet,'clf')
classif.classify_bycluster()

