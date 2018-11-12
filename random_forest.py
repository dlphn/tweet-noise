"""Ce modèle sur représente les spams. Trop de faux positif (1 au lieu de 0).
A voir si on peut donner un poid plus important aux données d'entrainement ou y == 0"""

# Required Python Packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns

from classification import *

dataset = df_tweets_categorized
HEADERS = ['nb_follower', 'nb_following', 'verified', 'reputation', 'age', 'nb_tweets', 'time', 'proportion_spamwords',
       'orthographe', 'RT', 'spam']


def split_dataset(dataset, train_percentage, feature_headers, target_header):
    # Split dataset into train and test dataset
    train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers], dataset[target_header],
                                                        train_size=train_percentage)
    return train_x, test_x, train_y, test_y

def random_forest_classifier(features, target):
    clf = RandomForestClassifier(class_weight={0:5,1:1})
    clf.fit(features, target)
    return clf

def randomtree():
    train_x, test_x, train_y, test_y = split_dataset(dataset, 0.7, HEADERS[1:-1], HEADERS[-1])
    trained_model = random_forest_classifier(train_x, train_y)
    #print("Trained model :: ", trained_model)
    predictions = trained_model.predict(test_x)
    print ("Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x)))
    print ("Test Accuracy  :: ", accuracy_score(test_y, predictions))
    cm = pd.DataFrame(confusion_matrix(test_y, predictions), columns=[0,1], index=[0,1])
    #sns.heatmap(cm, annot=True)
    print(cm)
    cmpt = 0
    for elt in test_y:
        if elt == 0:
            cmpt += 1
    print(cmpt)

randomtree()

