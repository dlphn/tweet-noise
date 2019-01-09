# Tweet Noise

[![Python](https://img.shields.io/badge/python-3.6-blue.svg?style=flat-square)](https://docs.python.org/3/)

> Spam detection on Twitter

## Description

Python school project developed by 3 students from CentraleSupélec : Hélène, Valentin, and Delphine.

## Installation

Add a `config.py` file which contains Twitter API and MongoDB keys.
```python
# OAuth authentification keys to Twitter API
ACCESS = (
    # consumer_key,
    # consumer_secret,
    # oauth_token,
    # oauth_token_secret
)

# Restricts tweets to the given language, given by an ISO 639-1 code.
# Language detection is best-effort.
# To fetch all tweets, choose LANG = None
LANG = 'fr'

# Directory where tweets files are stored
FILEDIR = str

# Number of tweets in each file
FILEBREAK = 1000

PROXY = {'http': '',
         'https': ''}

# MongoDB cluster config
MONGODB = {
    "USER": str,
    "PASSWORD": str,
    "HOST": str,
    "PORT": 27017,
    "DATABASE": str
}
```


## Project structure

- `streamingAPI.py`: fetch data from Twitter API and save in MongoDB
- `featuresBuilder.py`: fetch data from MongoDB and build the features table
- `JSONBuilder.py`: fetch data from MongoDB and build 2 JSON files for spam and info
- `spamKeywords.py`: list of key words considered as spam
- `dataLabelling.py`: small algorithm to ease the data labelling process
- `classification.py`: fetch the features tables and categorize the different features
- (`classification2.py`: duplicate of `classification.py`)
- Data Visualization :
    - `features_analysis.py` : matplotlib
    - `dataVisualization.py` : seaborn tests
    - `dataViz.py` : matplotlib tests
- Classification :
    - `IfClassification.py` : simple if/else classification
    - `scikit_classification.py` : test and compare different scikitlearn classifiers
    - `KNearest.py` : K Nearest Neighbours classifier
    - `random_forest.py` : Random Forest classifier
    - `Support_Vector_Machine.py` : Support Vector Machine classifier
- Classification (npm) `bayesnpm/` : implement a simple text classifier with the Bayes NPM package
- Clustering :
    - `textClustering.py` : text processing and tf-idf vectorizer fitted on a k-means model 
    - `clustering.py`
    - `clusteringTest.py`


