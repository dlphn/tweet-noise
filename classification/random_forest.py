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
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import datetime
import sys
sys.path.append('..')
from config import FILEDIR


"""Ce modèle sur représente les spams. Trop de faux positif (1 au lieu de 0).
j'ai donné un poids 5 fois plus important aux données d'entrainement où y == 0. Peu concluant..."""

dataset = pd.read_csv(FILEDIR+'newtypes_categorized.csv')

def split_dataset(dataset, train_percentage, feature_headers, target_header):
    # Split dataset into train and test dataset
    train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers], dataset[target_header],
                                                        train_size=train_percentage)
    return train_x, test_x, train_y, test_y

def random_forest_classifier(features, target):
    t_start = time.clock()
    clf = RandomForestClassifier(class_weight={1:10,2:10,5:1,6:1,7:1,8:1})
    clf.fit(features, target)
    t_end = time.clock()
    t_diff = t_end - t_start
    print("trained {c} in {f:.2f} s".format(c="Random Forest", f=t_diff))
    return clf

def randomtree(dataset):
    HEADERS = dataset.columns.values.tolist()
    train_x, test_x, train_y, test_y = split_dataset(dataset, 0.7, HEADERS[1:-2], HEADERS[-2])
    trained_model = random_forest_classifier(train_x, train_y)
    #print("Trained model :: ", trained_model)
    predictions = trained_model.predict(test_x)
    print(Score(test_y, predictions))

def Score(y, predicted_y):
    acc =accuracy_score(y, predicted_y)
    cm = pd.DataFrame(confusion_matrix(y, predicted_y), columns=[ 1,2,5,6,7,8], index=[1,2,5,6,7,8])
    print(cm)
    print(cm[1][8])
    alpha = cm[1][1]+cm[1][2]+cm[2][1]+cm[2][2]
    beta = cm[2][8]+cm[2][1]+cm[2][2]+cm[2][5] +cm[2][6]+cm[2][7]+cm[1][8]+cm[1][1]+cm[1][2]+cm[1][5] +cm[1][6]+cm[1][7]
    gamma = cm[8][2]+cm[1][2]+cm[2][2]+cm[5][2] +cm[6][2]+cm[7][2]+cm[8][1]+cm[1][1]+cm[2][1]+cm[5][1] +cm[6][1]+cm[7][1]
    precision = alpha/beta
    recall = alpha/gamma
    if precision == 0 and recall == 0 :
        F_score = 0
    else :
        F_score = 2*precision*recall / (precision+ recall)
    return "Precision = {} \n Recall = {} \n F_score ={} ".format(precision, recall, F_score)

randomtree(dataset)
#print('ok')
#randomtree(df)
