# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 10:55 2018

@author: dshi, hbaud, vlefranc
"""

from collections import Counter, defaultdict
import numpy as np


class NaiveBaseClass:
    def calculate_relative_occurrences(self, values):
        nb_values = len(values)
        ro_dict = dict(Counter(values))
        for key in ro_dict.keys():
            ro_dict[key] = ro_dict[key] / float(nb_values)
        return ro_dict

    def get_max_value_key(self, d1):
        values = d1.values()
        keys = d1.keys()
        max_value_index = values.index(max(values))
        max_key = keys[max_value_index]
        return max_key

    def initialize_nb_dict(self):
        self.nb_dict = {}
        for label in self.labels:
            self.nb_dict[label] = defaultdict(list)


class NaiveBayes(NaiveBaseClass):
    """
    Naive Bayes Classifier method:
    It is trained with a 2D-array X (dimensions m,n) and a 1D array Y (dimension 1,n).
    X should have one column per feature (total n) and one row per training example (total m).
    After training a hash table is filled with the class probabilities per feature.
    We start with an empty hash table nb_dict, which has the form:

    nb_dict = {
        'class1': {
            'feature1': [],
            'feature2': [],
            (...)
            'featuren': []
        }
        'class2': {
            'feature1': [],
            'feature2': [],
            (...)
            'featuren': []
        }
    }
    """

    def train(self, X, Y):
        self.labels = np.unique(Y)  # true/false
        nb_rows, nb_cols = np.shape(X)
        self.initialize_nb_dict(self.labels)
        self.class_probabilities = self.calculate_relative_occurrences(Y)
        # iterate over all classes
        for label in self.labels:
            # first we get a list of indices per class, so we can take a subset X_ of the matrix X, containing data of only that class.
            row_indices = np.where(Y == label)[0]
            X_ = X[row_indices, :]

            # in this subset, we iterate over all the columns/features, and add all values of each feature to the hash table nb_dict
            nb_rows_, nb_cols_ = np.shape(X_)
            for feature in range(0, nb_cols_):
                self.nb_dict[label][feature] += list(X_[:, feature])
        # Now we have a Hash table containing all occurences of feature values, per feature, per class
        # We need to transform this Hash table to a Hash table with relative feature value occurrences per class
        for label in self.labels:
            for feature in range(0, nb_cols):
                self.nb_dict[label][feature] = self.calculate_relative_occurrences(self.nb_dict[label][feature])

    def classify_single_elem(self, X_elem):
        Y_dict = {}
        for label in self.labels:
            class_probability = self.class_probabilities[label]
            for line in range(0, len(X_elem)):
                relative_feature_values = self.nb_dict[label][line]
                if X_elem[line] in relative_feature_values.keys():
                    class_probability *= relative_feature_values[X_elem[line]]
                else:
                    class_probability *= 0
            Y_dict[label] = class_probability
        return self.get_max_value_key(Y_dict)

    def classify(self, X):
        self.predicted_Y_values = []
        nb_rows, nb_cols = np.shape(X)
        for line in range(0, nb_rows):
            X_elem = X[line,:]
            prediction = self.classify_single_elem(X_elem)
            self.predicted_Y_values.append(prediction)
        return self.predicted_Y_values

####

# X_train, Y_train, X_test, Y_test
# print("training naive bayes")
# nbc = NaiveBayes()
# nbc.train(X_train, Y_train)
# print("trained")
# predicted_Y = nbc.classify(X_test)
# y_labels = np.unique(Y_test)
# for y_label in y_labels:
#     f1 = f1_score(predicted_Y, Y_test, y_label)
#     print("F1-score on the test-set for class %s is: %s" % (y_label, f1))
