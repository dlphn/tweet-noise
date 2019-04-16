import csv
import json
import random

# Open the CSV containing ids, category, label and not truncated text
# Modify f with the right path
f = open('/home/valentin/Documents/3A/OSY/TweetNoise/tweets_all_2019-03-06.csv', 'r')

reader = csv.DictReader(f)
rows = list(reader)
random.shuffle(rows)

train_data_multi = []
test_data_multi = []


#Separation of the dataset in train / test json
for i in range(len(rows)) :
    if i < len(rows)*0.8:
        train_data_multi.append({"class" : rows[i]["category"], "text" : rows[i]["text"]})
    else :

        test_data_multi.append({"class": rows[i]["category"], "text": rows[i]["text"]})

test_data_multi.sort()

with open('train_data_multi.json', 'w') as outfile:
    json.dump(train_data_multi, outfile)


with open('test_data_multi.json', 'w') as outfile:
    json.dump(test_data_multi, outfile)

