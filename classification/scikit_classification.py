# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 15:55 2018

@author: dshi, hbaud, vlefranc
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
from IPython.display import display

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from classification.classification2 import Classification

warnings.filterwarnings("ignore", category=FutureWarning)

# Categorized data frame
classif = Classification()
df_tweets_categorized = classif.create_dataframe()
K_value = 7
dict_classifiers = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(n_neighbors=K_value, weights='distance', algorithm='auto'),
    "Linear SVM": SVC(gamma='scale', class_weight={0: 5, 1: 1}, kernel='rbf'),
    "Random Forest": RandomForestClassifier(class_weight={0: 5, 1: 1}),
    "Naive Bayes": GaussianNB(),
    "LDA": LinearDiscriminantAnalysis(),
    # "CART": DecisionTreeClassifier()
}
dict_models = {}

tweets = df_tweets_categorized
HEADERS = df_tweets_categorized.columns.values.tolist()

seed = 7


def split_dataset(dataset, train_percentage, feature_headers, target_header):
    """
    Split dataset into train and test dataset
    """
    train_x, test_x, train_y, test_y = train_test_split(
        dataset[feature_headers],
        dataset[target_header],
        train_size=train_percentage)
    return train_x, test_x, train_y, test_y


def evaluate(y, predicted_y):
    acc = accuracy_score(y, predicted_y)
    cm = pd.DataFrame(confusion_matrix(y, predicted_y), columns=[0, 1], index=[0, 1])
    precision = cm[0][0] / (cm[0][0] + cm[0][1])
    recall = cm[0][0] / (cm[0][0] + cm[1][0])
    f_score = 2 * precision * recall / (precision + recall) if precision > 0 and recall > 0 else 0
    return acc, precision, recall, f_score
    # return "Precision = {} \nRecall = {} \nF score = {}".format(precision, recall, f_score)


def classify(classifier_name, classifier, train_x, test_x, train_y, test_y, verbose=True):
    t_start = time.time()
    classifier.fit(train_x, train_y)
    t_end = time.time()

    t_diff = t_end - t_start
    train_score = classifier.score(train_x, train_y)
    test_score = classifier.score(test_x, test_y)

    dict_models[classifier_name] = {
        'model': classifier,
        'train_score': train_score,
        'test_score': test_score,
        'train_time': t_diff
    }

    # if verbose:
    #     print("trained {c} in {f:.2f} s".format(c=classifier_name, f=t_diff))
    return classifier


def display_dict_models(dict, sort_by='test_score'):
    cls = [key for key in dict.keys()]
    test_s = [dict[key]['test_score'] for key in cls]
    training_s = [dict[key]['train_score'] for key in cls]
    training_t = [dict[key]['train_time'] for key in cls]
    accuracy = [dict[key]['accuracy'] for key in cls]
    precision = [dict[key]['precision'] for key in cls]
    recall = [dict[key]['recall'] for key in cls]
    f_score = [dict[key]['f_score'] for key in cls]

    columns = ['classifier', 'train_score', 'test_score', 'train_time', 'accuracy', 'precision', 'recall', 'f_score']

    df_ = pd.DataFrame(data=np.zeros(shape=(len(cls), len(columns))),
                       columns=columns)
    for ii in range(0, len(cls)):
        df_.loc[ii, 'classifier'] = cls[ii]
        df_.loc[ii, 'train_score'] = training_s[ii]
        df_.loc[ii, 'test_score'] = test_s[ii]
        df_.loc[ii, 'train_time'] = training_t[ii]
        df_.loc[ii, 'accuracy'] = accuracy[ii]
        df_.loc[ii, 'precision'] = precision[ii]
        df_.loc[ii, 'recall'] = recall[ii]
        df_.loc[ii, 'f_score'] = f_score[ii]

    display(df_.sort_values(by=sort_by, ascending=False))


def predict(dataset, classifier):
    train_x, test_x, train_y, test_y = split_dataset(dataset, 0.7, HEADERS[1:-1], HEADERS[-1])
    clf = classify(classifier, dict_classifiers[classifier], train_x, test_x, train_y, test_y)
    pred_y = clf.predict(test_x)
    acc, precision, recall, f_score = evaluate(test_y, pred_y)
    dict_models[classifier_name]['accuracy'] = acc
    dict_models[classifier_name]['precision'] = precision
    dict_models[classifier_name]['recall'] = recall
    dict_models[classifier_name]['f_score'] = f_score


def compare(dataset):
    """
    Evaluate each model in turn
    """
    results = []
    names = []
    scoring = 'accuracy'
    train_x, test_x, train_y, test_y = split_dataset(dataset, 0.7, HEADERS[1:-1], HEADERS[-1])
    for name in dict_classifiers.keys():
        model = dict_classifiers[name]
        kfold = KFold(n_splits=10, random_state=seed)
        cv_results = cross_val_score(model, train_x, train_y, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()


for classifier_name in dict_classifiers.keys():
    predict(tweets, classifier_name)

print("\n================================================= Results ================================================= ")
display_dict_models(dict_models)

print()
compare(tweets)
