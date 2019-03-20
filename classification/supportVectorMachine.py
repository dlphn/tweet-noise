# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 15:55 2018

@author: dshi, hbaud, vlefranc
"""
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split


from classification.classification import Classification


classif = Classification()
dataset = classif.create_dataframe()
#print(dataset.describe())
HEADERS = ['nb_follower', 'nb_following', 'verified', 'reputation', 'age', 'nb_tweets', 'time', 'proportion_spamwords',
       'orthographe', 'RT', 'spam']

def split_dataset(dataset, train_percentage, feature_headers, target_header):
    # Split dataset into train and test dataset
    train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers], dataset[target_header],
                                                        train_size=train_percentage)
    return train_x, test_x, train_y, test_y

def Support_Vector_Machine_Classifier(train_x, train_y):
    clf = svm.SVC(gamma='scale',class_weight={0:5,1:1}, kernel='rbf')
    clf.fit(train_x, train_y)
    return clf

def SVM():
    train_x, test_x, train_y, test_y = split_dataset(dataset, 0.7, HEADERS[1:-1], HEADERS[-1])
    #print(train_y)
    clf = Support_Vector_Machine_Classifier(train_x,train_y)
    pred_y = clf.predict(test_x)
    print(Score(test_y, pred_y))

def Score(y, predicted_y):
    acc =accuracy_score(y, predicted_y)
    cm = pd.DataFrame(confusion_matrix(y, predicted_y), columns=[0, 1], index=[0, 1])
    precision = cm[0][0]/(cm[0][0]+cm[0][1])
    recall = cm[0][0]/(cm[0][0]+cm[1][0])
    F_score = 2*precision*recall / (precision+ recall)
    return "Precision = {} \n Recall = {} \n F_score ={} ".format(precision, recall, F_score)

print(SVM())
