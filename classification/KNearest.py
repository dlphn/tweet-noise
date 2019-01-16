# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 10:55 2018

@author: dshi, hbaud, vlefranc
"""

"""Weight de l'algorithm uniform ou distance, à voir --> faire bcp de repetition pour voir le meilleur score.
Toujours un problème de faux positif !"""

from math import *
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
from features_analysis import PCA

from classification.classification2 import Classification

classif = Classification()
dataset = classif.create_dataframe()
HEADERS = ['nb_follower', 'nb_following', 'verified', 'reputation', 'age', 'nb_tweets', 'time', 'proportion_spamwords',
       'orthographe', 'RT', 'spam']


def euclidean_distance(x, y):
    return sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))

def split_dataset(dataset, train_percentage, feature_headers, target_header):
    # Split dataset into train and test dataset
    train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers], dataset[target_header],
                                                        train_size=train_percentage)
    return train_x, test_x, train_y, test_y

def Score(y, predicted_y):
    acc =accuracy_score(y, predicted_y)
    cm = pd.DataFrame(confusion_matrix(y, predicted_y), columns=[0, 1], index=[0, 1])
    precision = cm[0][0]/(cm[0][0]+cm[0][1])
    recall = cm[0][0]/(cm[0][0]+cm[1][0])
    if precision == 0 and recall == 0 :
        F_score = 50
    else :
        F_score = 2*precision*recall / (precision+ recall)
    return F_score

def K_Neighbors_classifier(K_value, train_x, train_y):
    neigh = KNeighborsClassifier(n_neighbors=K_value, weights='distance', algorithm='auto')
    neigh.fit(train_x, train_y)
    return neigh


def KNeighbors (dataset):
    y_plot=[]
    for K in range(10):
        K_value = K+1
        r =0
        for i in range(100):
            train_x, test_x, train_y, test_y = split_dataset(dataset, 0.7,['principal component 1','principal component 2'],['spam'])
            #print(val_y)
            neigh = K_Neighbors_classifier(K_value, train_x, train_y)
            pred_y = neigh.predict(test_x)
            r = r + Score(test_y,pred_y)/100
        y_plot.append(r)
        #print(y_plot)
    plt.plot(y_plot)
    plt.axis(ymax=0.3, ymin =0.1)
    plt.show()


dataset_pca = PCA(dataset.iloc[:,:-1],dataset.iloc[:,-1])
print(dataset_pca.head())
KNeighbors(dataset_pca)

"""
def choose_features ():
    cm_max = 0
    acc_max = 0
    (a,b) = (0,0)
    (c,d) = (0,0)
    for i in range (1,len(HEADERS)-1):
        for j in range (i) :
            train_x, test_x, train_y, test_y = split_dataset(dataset, 0.7, HEADERS[j:i], HEADERS[-1])
            #print(train_x.shape)
            acc_sum = 0
            cm_sum = 0
            for k in range (20):
                neigh = K_Neighbors_classifier(int(k//5)+1, train_x, train_y)
                pred_y = neigh.predict(test_x)
                cm = pd.DataFrame(confusion_matrix(test_y, pred_y), columns=[0, 1], index=[0, 1])
                acc_sum = acc_sum + accuracy_score(test_y,pred_y)
                cm_sum = cm_sum + cm[0][0]/(cm[1][0]+cm[0][0])
            if acc_sum/20 > acc_max :
                acc_max = acc_sum/20
                (a,b)=(j,i)
            if cm_sum/20 > cm_max :
                cm_max = cm_sum/20
                print(cm_max)
                (c,d)=(j,i)
    print (acc_max, (a,b),cm_max, (c,d) )"""