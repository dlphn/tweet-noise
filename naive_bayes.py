# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 10:55 2018

@author: dshi, hbaud, vlefranc
"""

from collections import Counter, defaultdict
import numpy as np


class NaiveBaseClass:
    def calculate_relative_occurences(self, list):
        no_examples = len(list)
        ro_dict = dict(Counter(list))
        for key in ro_dict.keys():
            ro_dict[key] = ro_dict[key] / float(no_examples)
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
        self.labels = np.unique(Y)
        no_rows, no_cols = np.shape(X)
        self.initialize_nb_dict(self.labels)
        self.class_probabilities = self.calculate_relative_occurences(Y)
