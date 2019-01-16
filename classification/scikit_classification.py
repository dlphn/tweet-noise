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

from classification.classification2 import Classification


#Data frame classifié
classif = Classification()
df_tweets_categorized = classif.create_dataframe()
K_value = 7
dict_classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Nearest Neighbors": KNeighborsClassifier(n_neighbors=K_value, weights='distance', algorithm='auto'),
    "Linear SVM": SVC(gamma='scale', class_weight={0: 5, 1: 1},kernel='rbf'),
    "Random Forest": RandomForestClassifier(class_weight={0: 5, 1: 1}),
    "Naive Bayes": GaussianNB(),
}
dict_models = {}

tweets = df_tweets_categorized
HEADERS = df_tweets_categorized.columns.values.tolist()


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
    F_score = 2*precision*recall / (precision+ recall)
    return "Precision = {} \n Recall = {} \n F_score ={}".format(precision, recall, F_score)


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


classif =[]


def predict(dataset, classifier):
    train_x, test_x, train_y, test_y = split_dataset(dataset, 0.7, HEADERS[1:-1], HEADERS[-1])
    clf = classify(classifier, dict_classifiers[classifier], train_x, test_x, train_y, test_y)
    pred_y = clf.predict(test_x)
    score = Score(test_y,pred_y)
    classif.append(classifier)
    classif.append(score)


def display_classif():
    for i in range(len(classif)):
        print(classif[i] + " \n")


predict(tweets, "Naive Bayes")
predict(tweets, "Linear SVM")
predict(tweets, "Nearest Neighbors")
predict(tweets, "Random Forest")
print("\n========================= Results ========================= ")
display_classif()
