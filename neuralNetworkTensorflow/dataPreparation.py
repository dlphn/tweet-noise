import csv
import json
import random

# Open the CSV
f = open('/home/valentin/Documents/3A/OSY/TweetNoise/tweets_all_2019-03-06.csv', 'r')

reader = csv.DictReader(f)

data = {}
actualite = []
reaction = []
conversation = []
pub = []
bot = []
other = []

rows = list(reader)

validation = []
category = []
text = []

random.shuffle(rows)

for i in range(len(rows)) :
    if i < len(rows)*0.8:
        if rows[i]["category"] == "actualité" :
            actualite.append(rows[i]["text"])
        elif rows[i]["category"] == "reaction" :
            reaction.append(rows[i]["text"])
        elif rows[i]["category"] == "conversation" :
            conversation.append(rows[i]["text"])
        elif rows[i]["category"] == "publicité" :
            pub.append(rows[i]["text"])
        elif rows[i]["category"] == "bot" :
            bot.append(rows[i]["text"])
        elif rows[i]["category"] == "other spam" :
            other.append(rows[i]["text"])
    else :
        category.append(rows[i]["category"])
        text.append(rows[i]["text"])


data["actualité"] = actualite
data["reaction"] = reaction
data["conversation"] = conversation
data["publicité"] = pub
data["bot"] = bot
data["other spam"] = other

validation.append(category)
validation.append(text)

with open('data.json', 'w') as outfile:
    json.dump(data, outfile)
