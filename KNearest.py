# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 10:55 2018

@author: dshi, hbaud, vlefranc
"""

"""Weight de l'algorithm uniform ou distance, à voir --> faire bcp de repetition pour voir le meilleur score.
Toujours un problème de faux positif !"""

from math import *
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

from classification import *

dataset = df_tweets_categorized
HEADERS = ['nb_follower', 'nb_following', 'verified', 'reputation', 'age', 'nb_tweets', 'time', 'proportion_spamwords',
       'orthographe', 'RT', 'spam']


def euclidean_distance(x, y):
    return sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))

def split_dataset(dataset, train_percentage, feature_headers, target_header):
    # Split dataset into train and test dataset
    train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers], dataset[target_header],
                                                        train_size=train_percentage)
    return train_x, test_x, train_y, test_y

def K_Neighbors_classifier(K_value, train_x, train_y):
    neigh = KNeighborsClassifier(n_neighbors=K_value, weights='distance', algorithm='auto')
    neigh.fit(train_x, train_y)
    return neigh


def KNeighbors (n):
    train_x, test_x, train_y, test_y = split_dataset(dataset, 0.7, HEADERS[1:-1], HEADERS[-1])
    for K in range(n):
        K_value = K+1
        neigh = K_Neighbors_classifier(K_value, train_x, train_y)
        pred_y = neigh.predict(test_x)
        cm = pd.DataFrame(confusion_matrix(test_y, pred_y), columns=[0, 1], index=[0, 1])
        print ("Accuracy is ", accuracy_score(test_y,pred_y)*100,"% for K-Value:",K_value,
               "\nActu catégorisées Actu = {}, Actu catégorisées Spam={}".format(cm[0][0],cm[1][0]))


KNeighbors(7)
