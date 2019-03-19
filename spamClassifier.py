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
        #print(type(self.tweet))
        num_cluster = self.tweet['pred']
        #print(num_cluster)
        if num_cluster == -1:
            cluster_class, rf_class = self.set_cluster_spam(True)
            return cluster_class,rf_class
        else :
            df_pred = self.df[self.df['pred']== num_cluster]
            length = df_pred.shape[0]
            if length < 4 :
                cluster_class, rf_class = self.set_cluster_spam(False)
                return cluster_class,rf_class
            else :
                df_pred.label.apply(self.categorize_label)
                if df_pred.label.sum(axis = 0)/length > 0.1 :
                    cluster_class, rf_class = self.set_cluster_spam(False)
                    return cluster_class,rf_class
                else :
                    if self.contains_media(df_pred) :
                        cluster_class, rf_class = self.set_cluster_spam(False)
                        return cluster_class,rf_class
                    else :
                        cluster_class, rf_class = self.set_cluster_spam(True)
                        return cluster_class,rf_class


    def set_cluster_spam(self, bool):
        if bool:
            cluster_class = '0'
            rf_class = 'undefined'
            return cluster_class,rf_class
        else:
            cluster_class = 'undefined'
            rf_class = str(self.classify_byrandomforest())
            return cluster_class,rf_class

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





