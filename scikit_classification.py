# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 15:55 2018

@author: dshi, hbaud, vlefranc
"""
import pandas as pd
import numpy as np
import time
from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

from classification import df_tweets_categorized

K_value = 7
dict_classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Nearest Neighbors": KNeighborsClassifier(n_neighbors=K_value, weights='distance', algorithm='auto'),
    "Linear SVM": SVC(gamma='scale', class_weight={0: 10, 1: 1}),
    "Random Forest": RandomForestClassifier(n_estimators=1000),
    "Naive Bayes": GaussianNB(),
}
dict_models = {}

tweets = df_tweets_categorized
HEADERS = ['nb_follower', 'nb_following', 'verified', 'reputation', 'age', 'nb_tweets', 'time', 'proportion_spamwords',
       'orthographe', 'RT', 'spam']


def split_dataset(dataset, train_percentage, feature_headers, target_header):
    # Split dataset into train and test dataset
    train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers], dataset[target_header],
                                                        train_size=train_percentage)
    return train_x, test_x, train_y, test_y


def classify(classifier_name, classifier, train_x, test_x, train_y, test_y, verbose=True):
    t_start = time.clock()
    classifier.fit(train_x, train_y)
    t_end = time.clock()

    t_diff = t_end - t_start
    train_score = classifier.score(train_x, train_y)
    test_score = classifier.score(test_x, test_y)

    dict_models[classifier_name] = {'model': classifier, 'train_score': train_score, 'test_score': test_score,
                                    'train_time': t_diff}

    if verbose:
        print("trained {c} in {f:.2f} s".format(c=classifier_name, f=t_diff))
    return classifier


def display_dict_models(dict, sort_by='test_score'):
    cls = [key for key in dict.keys()]
    test_s = [dict[key]['test_score'] for key in cls]
    training_s = [dict[key]['train_score'] for key in cls]
    training_t = [dict[key]['train_time'] for key in cls]

    df_ = pd.DataFrame(data=np.zeros(shape=(len(cls), 4)),
                       columns=['classifier', 'train_score', 'test_score', 'train_time'])
    for ii in range(0, len(cls)):
        df_.loc[ii, 'classifier'] = cls[ii]
        df_.loc[ii, 'train_score'] = training_s[ii]
        df_.loc[ii, 'test_score'] = test_s[ii]
        df_.loc[ii, 'train_time'] = training_t[ii]

    display(df_.sort_values(by=sort_by, ascending=False))


def predict(dataset, classifier):
    train_x, test_x, train_y, test_y = split_dataset(dataset, 0.7, HEADERS[1:-1], HEADERS[-1])
    clf = classify(classifier, dict_classifiers[classifier], train_x, test_x, train_y, test_y)
    pred_y = clf.predict(test_x)
    cm = pd.DataFrame(confusion_matrix(test_y, pred_y), columns=[0, 1], index=[0, 1])
    print("Accuracy (test score) is ", accuracy_score(test_y, pred_y) * 100,
          "\nActu catégorisées Actu = {}, Actu catégorisées Spam={}. Ratio={} ".format(cm[0][0], cm[1][0], cm[0][0]/(cm[0][0]+cm[1][0])))
    print(cm)


predict(tweets, "Naive Bayes")
predict(tweets, "Linear SVM")
predict(tweets, "Nearest Neighbors")
predict(tweets, "Random Forest")
display_dict_models(dict_models)