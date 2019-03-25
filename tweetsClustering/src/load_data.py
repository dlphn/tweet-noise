# -*- coding: utf-8 -*-
"""
@author: bmazoyer

edits by dshi, hbaud, vlefranc
"""


# import logging
import csv
# from datetime import datetime, timedelta
import re
# import pickle
import pandas as pd
# from elasticsearch import Elasticsearch, helpers
from tweetsClustering.docs.config import ES_DATE_FORMAT, ELASTIC, DATA_PATH
from unidecode import unidecode
# import os

from string import punctuation
punctuation_fr = "«»…" + punctuation


def must_not():
    # returns a list of twitter sources that should be deleted (porn, videogames, bots)
    must_not = []
    with open(DATA_PATH + "sources_twitter.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row[3] == "0":
                must_not.append({"term": {"source.keyword": row[0]}})
    return must_not


def remove_repeated_characters(expr):
    # limit number of repeated letters to 3. For example loooool --> loool
    string_not_repeated = ""
    for item in re.findall(r"((.)\2*)", expr):
        if len(item[0]) <= 3:
            string_not_repeated += item[0]
        else:
            string_not_repeated += item[0][:3]
    return string_not_repeated


def camel_case_split(expr):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', expr)
    return " ".join([m.group(0) for m in matches])


def format_text(text):

    # translate to equivalent ascii characters and erase urls
    try:
        text = unidecode(re.sub(r"http\S+", '', text, flags=re.MULTILINE))
        text = unidecode(re.sub(r"@\S+", '', text, flags=re.MULTILINE))
    except Exception:
        print(text)
        raise
    new_text = []

    for word in re.split(r"[' ]", text):
        # remove numbers longer than 4 digits
        if len(word) < 5 or not word.isdigit():
            if word.startswith("#"):
                # add lowered hashtags to have them 2 times (alexandrebenalla and Alexandre Benalla)
                #new_text.append(word[1:].lower())
                new_text.append(camel_case_split(word[1:]))
            else:
                new_text.append(word)
    text = remove_repeated_characters(" ".join(new_text))
    return text


def format_docs(doc):
    if "extended_tweet" in doc and "full_text" in doc["extended_tweet"]:
        text = format_text(doc["extended_tweet"]["full_text"])
    else:
        text = format_text(doc["text"])
    if "quoted_status_text" in doc:
         text = text + " " + format_text(doc["quoted_status_text"])

    in_event = {}
    if "human" in doc:  # or "machine" in doc:
        annotation = doc["human"]["annotated"] if "human" in doc else doc["machine"]["annotated"]
        for user in annotation:
            for event in annotation[user]:
                if event not in in_event:
                    in_event[event] = True
                in_event[event] = in_event[event] * annotation[user][event]["in_event"]

    label = None
    for event in in_event:
        if in_event[event]:
            label = event

    return {"text": text, "id": doc["id_str"], "label": label}


def body(start, end, with_retweets):
    body = {
        "query": {
            "bool": {
                "filter": [
                    {"term": {"lang": "fr"}},
                    {"range": {"created_at": {
                        "gte": start,
                        "lte": end
                    }}}
                ],
                "must_not": must_not()
            }
        }
    }
    if with_retweets == False:
        body["query"]["bool"]["filter"].append({"term": {"is_retweet": False}})
    return body


def load_data(path):
    data = pd.read_csv(path, dtype={"label": "str", "category": "str", "id": "str", "text": "str"})
    data = data.fillna("")
    data["text"] = data.text.map(format_text)
    return data[["id", "text", "label", "category"]].sort_values("id").reset_index(drop=True)


def load_all_data(path):
    data = pd.read_csv(path, dtype={
        "id": "str",
        "label": "str",
        "category": "str",
        "text": "str",
    })
    data = data.fillna("")
    data["text"] = data.text.map(format_text)
    return data[['id', 'label', 'category', 'text', 'screen_name', 'nb_follower', 'nb_following', 'verified',
                 'reputation', 'age', 'nb_tweets', 'posted_at', 'length', 'proportion_spamwords', 'orthographe',
                 'nb_hashtag', 'nb_urls', 'nb_emoji']].sort_values("id").reset_index(drop=True)


if __name__ == "__main__":
    text = "Mais alors @EPhilippePM devrait avoir à cœur de dénoncer @TeamMacronPR pour avoir " \
           "délibérément fabriqué une FakeNews #SalutNazi Hypocrisie suprême c’est la Rapporteuse de la manipulation " \
           "de l’information @NaimaMoutchou qui colporte la #FakeNews 🤣"
    print(format_text(text))
