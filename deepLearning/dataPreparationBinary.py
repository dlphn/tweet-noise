import csv
import json
import random

# Open the CSV containing ids, category, label and not truncated text
# Modify f with the right path
f = open('/home/valentin/Documents/3A/OSY/TweetNoise/tweets_all_2019-03-06.csv', 'r')

reader = csv.DictReader(f)
rows = list(reader)
random.shuffle(rows)

train_data_spam = []
train_data_actualite = []
test_data_spam = []
test_data_actualite = []

#Separation of the dataset in train / test json and spam/actuality
for i in range(len(rows)) :
    if i < len(rows)*0.8:
        if rows[i]["label"] == "actualité" :
            train_data_actualite.append(rows[i]["text"])
        elif rows[i]["label"] == "spam" :
            train_data_spam.append(rows[i]["text"])

    else :
        if rows[i]["label"] == "actualité":
            test_data_actualite.append(rows[i]["text"])
        elif rows[i]["label"] == "spam":
            test_data_spam.append(rows[i]["text"])

with open('train_data_spam.json', 'w') as outfile:
    json.dump(train_data_spam, outfile)

with open('train_data_actualite.json', 'w') as outfile:
    json.dump(train_data_actualite, outfile)

with open('test_data_spam.json', 'w') as outfile:
    json.dump(test_data_spam, outfile)

with open('test_data_actualite.json', 'w') as outfile:
    json.dump(test_data_actualite, outfile)
