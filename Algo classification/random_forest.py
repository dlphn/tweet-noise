# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 10:55 2018

@author: dshi, hbaud, vlefranc
"""

# Required Python Packages
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

from classification2 import Classification



"""Ce modèle sur représente les spams. Trop de faux positif (1 au lieu de 0).
j'ai donné un poids 5 fois plus important aux données d'entrainement où y == 0. Peu concluant..."""
classif = Classification()
dataset = classif.create_dataframe()
df2 = dataset[['nb_follower', 'nb_following', 'verified', 'nb_tweets', 'proportion_spamwords', 'proportion_whitewords',  'orthographe',  'nb_emoji','spam']]



def split_dataset(dataset, train_percentage, feature_headers, target_header):
    # Split dataset into train and test dataset
    train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers], dataset[target_header],
                                                        train_size=train_percentage)
    return train_x, test_x, train_y, test_y

def random_forest_classifier(features, target):
    t_start = time.clock()
    clf = RandomForestClassifier(class_weight={0:5,1:1})
    clf.fit(features, target)
    t_end = time.clock()
    t_diff = t_end - t_start
    print("trained {c} in {f:.2f} s".format(c="Random Forest", f=t_diff))
    return clf

def randomtree(dataset):
    HEADERS = dataset.columns.values.tolist()
    train_x, test_x, train_y, test_y = split_dataset(dataset, 0.7, HEADERS[1:-1], HEADERS[-1])
    trained_model = random_forest_classifier(train_x, train_y)
    #print("Trained model :: ", trained_model)
    predictions = trained_model.predict(test_x)
    print(Score(test_y, predictions))

def Score(y, predicted_y):
    acc =accuracy_score(y, predicted_y)
    cm = pd.DataFrame(confusion_matrix(y, predicted_y), columns=[0, 1], index=[0, 1])
    precision = cm[0][0]/(cm[0][0]+cm[0][1])
    recall = cm[0][0]/(cm[0][0]+cm[1][0])
    F_score = 2*precision*recall / (precision+ recall)
    print(cm)
    return "Precision = {} \n Recall = {} \n F_score ={} ".format(precision, recall, F_score)


randomtree(df2)

