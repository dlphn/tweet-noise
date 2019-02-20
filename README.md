# Tweet Noise

[![Python](https://img.shields.io/badge/python-3.6-blue.svg?style=flat-square)](https://docs.python.org/3/)

> Spam detection on Twitter

## Description

Python school project developed by 3 students from CentraleSupélec : Hélène, Valentin, and Delphine.

We consider as spam anything that is not related to a news event or a reaction to one.

## Installation

Install dependencies using pip (preferably in a virtual environment):

```bash
pip install -r requirements.txt
``

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
- `arrayBuilder.py`: fetch data from MongoDB and build an array of all the tweets (text and label only)
- `spamKeywords.py`: list of key words considered as spam
- `dataLabelling.py`: small algorithm to ease the data labelling process
- `classification.py`: fetch the features tables and categorize the different features
- Data Visualization :
    - `features_analysis.py` : matplotlib and seaborn
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
    - `clusteringTest.py` : tests [to be removed]


## Steps

### 1. Data Acquisition

Fetch data from Twitter API.

Run `streamingAPI.py`.

### 2. Data Preprocessing

Build features file.

Run `features.featuresBuilder.py`.

### 3. Visualization

Run `features_analysis.py`.

### 4. Classification

Run `classification.scikit_classification.py` to compare different scikit-learn classifiers.

### 5. Clustering

Run `testClustering.py`.

## Features stored in database

\* : used for analysis

° : to be considered


- _id
- created_at : str
- id : int
- id_str : str *
- text : str *
- source : str
- truncated : bool
- in_reply_to_status_id
- in_reply_to_status_id_str
- in_reply_to_user_id
- in_reply_to_user_id_str
- in_reply_to_screen_name
- user : object
    - id : int
    - id_str : str
    - name : str
    - screen_name : str
    - location : str
    - url : str
    - description : str
    - translator_type : str
    - protected : bool
    - verified : bool *
    - followers_count : int *
    - friends_count : int *
    - listed_count : int
    - favourites_count : int
    - statuses_count : int *
    - created_at : str *
    - utc_offset
    - time_zone
    - geo_enabled : bool
    - lang : str
    - contributors_enabled : bool
    - is_translator : bool
    - profile_background_color : str
    - profile_background_image_url : str
    - profile_background_image_url_https : str
    - profile_background_tile : bool
    - profile_link_color : str
    - profile_sidebar_border_color : str
    - profile_sidebar_fill_color : str
    - profile_text_color : str
    - profile_use_background_image : bool
    - profile_image_url : str
    - profile_image_url_https : str
    - profile_banner_url : str
    - default_profile : bool
    - default_profile_image : bool
    - following
    - follow_request_sent
    - notifications
- geo
- coordinates
- place
- contributors
- is_quote_status : bool
- extended_tweet : object
    - full_text : str
    - display_text_range : list
    - entities : object
        - hashtags : list
        - urls : list
        - user_mentions : list
        - symbols : list
        - media : list
    - extended_entities : object
        - media : list
- quote_count : int
- reply_count : int
- retweet_count : int
- favorite_count : int
- entities : object °
    - hashtags : list *
        - text : str
        - indices : list
    - urls : list
        - url : str
        - extended_url : str
        - display_irl : str
        - indices : list
    - user_mentions : list
        - screen_name : str
        - name : str
        - id : int
        - id_str : str
        - indices : list
    - symbols : list
    - media : list
- favorited : bool
- retweeted : bool
- possibly_sensitive : bool
- filter_level : str
- lang : str
- timestamp_ms : str *
- spam : bool *
- type : str *

