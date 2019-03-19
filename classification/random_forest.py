# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 10:55 2018

@author: dshi, hbaud, vlefranc
"""

# Required Python Packages
import time
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import seaborn as sns
import datetime
import sys
sys.path.append('..')
from config import FILEDIR

dataset = pd.read_csv(FILEDIR+'tweets_data2_categorized_spam.csv')

def train_model(dataset) :
    #print(dataset[dataset['nb_follower','verified']])
    #print(dataset.dtypes)
    dataset.label = dataset.label.apply(categorize_label)
    trained_model = random_forest_classifier(dataset[['nb_follower', 'nb_following', 'verified', 'reputation', 'age',
                                                     'nb_tweets', 'length', 'proportion_spamwords',
                                                     'orthographe', 'nb_hashtag', 'nb_urls', 'nb_emoji']], dataset['label'])
    return trained_model

def predict_rf(trained_model,tweet):
    tweet = tweet[['nb_follower', 'nb_following', 'verified', 'reputation', 'age','nb_tweets',
                        'length', 'proportion_spamwords','orthographe', 'nb_hashtag', 'nb_urls', 'nb_emoji']]
    tweet = tweet.tolist()
    prediction = trained_model.predict([tweet])
    return prediction[0]


def random_forest_classifier( features, target):
    t_start = time.clock()
    clf = RandomForestClassifier(class_weight= {0: 1, 1: 5}, max_depth= 10, min_samples_leaf= 5, min_samples_split= 20)
    #print(features.head())
    #print(target.head())
    clf.fit(features, target)
    t_end = time.clock()
    t_diff = t_end - t_start
    #print("trained {c} in {f:.2f} s".format(c="Random Forest", f=t_diff))
    return clf

def split_dataset(dataset, train_percentage, feature_headers, target_header):
    # Split dataset into train and test dataset
    train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers], dataset[target_header],
                                                        train_size=train_percentage)
    return train_x, test_x, train_y, test_y

def randomtree(dataset):
    HEADERS = dataset.columns.values.tolist()
    #print(HEADERS)
    train_x, test_x, train_y, test_y = split_dataset(dataset, 0.7, [ 'nb_follower', 'nb_following', 'verified', 'reputation', 'age', 'nb_tweets', 'posted_at', 'length', 'proportion_spamwords', 'orthographe', 'nb_hashtag', 'nb_urls', 'nb_emoji', 'type']
, HEADERS[-2])
    #param = gridsearch_rf(train_x, train_y)
    #print(param)
    print(train_x.head())
    trained_model = random_forest_classifier(train_x.drop('type',axis=1), train_y)
    predictions = trained_model.predict(test_x.drop('type',axis=1))
    test_x['prediction'] = predictions
    print(Score_spam(test_y, predictions))
    return test_x

def Score_type(y, predicted_y):
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

def Score_spam(y, predicted_y):
    acc =accuracy_score(y, predicted_y)
    cm = pd.DataFrame(confusion_matrix(y, predicted_y), columns=[ 0,1], index=[0,1])
    precision =cm[1][1]/(cm[1][0]+cm[1][1])
    recall = cm[1][1]/(cm[0][1]+cm[1][1])
    print(cm)
    if precision == 0 and recall == 0 :
        F_score = 0
    else :
        F_score = 2*precision*recall / (precision+ recall)
    return "Precision = {} \n Recall = {} \n F_score ={} ".format(precision, recall, F_score)

def gridsearch_rf(train_x,train_y):
    #fonction pour trouver les parametres a choisir pour optimiser le score de notre random forest
    max_depth = [10, 15, 20, None]
    min_samples_split = [ 5, 10, 20, 30, 45]
    min_samples_leaf = [2, 5, 10, 15]
    class_weight = [{0:1,1:5},{0:1,1:10},{0:1,1:15}]
    param_grid = {'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                  'class_weight' : class_weight}

    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='f1',
                               cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(train_x, train_y)
    return grid_search.best_params_

def categorize_label(x):
    if x == 'spam':
        return 0
    else:
        return 1




