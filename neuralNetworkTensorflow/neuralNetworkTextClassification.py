import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import json
import string
import unicodedata
import sys
from neuralNetworkTensorflow.dataPreparation import validation

stemmer = LancasterStemmer()

# a table structure to hold the different punctuation used
tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                    if unicodedata.category(chr(i)).startswith('P'))

# method to remove punctuations from sentences.
def remove_punctuation(text):
    return text.translate(tbl)

# initialize the stemmer
# variable to hold the Json data read from the file
data = None

# read the json file and load the training data
with open('data.json') as json_data:
    data = json.load(json_data)

# get a list of all categories to train for
categories = list(data.keys())
words = []
# a list of tuples with words in the sentence and category name
docs = []

for each_category in data.keys():
    for each_sentence in data[each_category]:
        # remove any punctuation from the sentence
        each_sentence = remove_punctuation(each_sentence)
        #print(each_sentence)
        # extract words from each sentence and append to the word list
        w = nltk.word_tokenize(each_sentence)
        #print("tokenized words: ", w)
        words.extend(w)
        docs.append((w, each_category))

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words]
words = sorted(list(set(words)))

# create our training data
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(categories)


for doc in docs:
    # initialize our bag of words(bow) for each document in the list
    bow = []
    # list of tokenized words for the pattern
    token_words = doc[0]
    # stem each word
    token_words = [stemmer.stem(word.lower()) for word in token_words]
    # create our bag of words array
    for w in words:
        bow.append(1) if w in token_words else bow.append(0)

    output_row = list(output_empty)
    output_row[categories.index(doc[1])] = 1

    # our training set will contain a the bag of words model and the output row that tells
    # which catefory that bow belongs to.
    training.append([bow, output_row])

# shuffle our features and turn into np.array as tensorflow  takes in numpy array
random.shuffle(training)
training = np.array(training)

# trainX contains the Bag of words and train_y contains the label/ category
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# reset underlying graph data
tf.reset_default_graph()

# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

def get_tf_record(sentence):
    global words
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    # bag of words
    bow = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bow[i] = 1
    return(np.array(bow))

if __name__ == '__main__':
    model.fit(train_x, train_y, n_epoch=500, batch_size=8, show_metric=True)
    model.save('model.tflearn')

validation_predicted = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
validation_actual = [0, 0, 0, 0, 0, 0]
for i in range (len(validation[0])) :
    test_i = categories[np.argmax(model.predict([get_tf_record(validation[1][i])]))]
    if validation[0][i] == "actualité" :
        validation_actual[0] += 1
        if test_i == "actualité":
            validation_predicted[0][0] += 1
        if test_i == "reaction":
            validation_predicted[0][1] += 1
        if test_i == "conversation":
            validation_predicted[0][2] += 1
        if test_i == "publicité":
            validation_predicted[0][3] += 1
        if test_i == "bot":
            validation_predicted[0][4] += 1
        if test_i == "other spam":
            validation_predicted[0][5] += 1
    elif validation[0][i] == "reaction":
        validation_actual[1] += 1
        if test_i == "actualité":
            validation_predicted[1][0] += 1
        if test_i == "reaction":
            validation_predicted[1][1] += 1
        if test_i == "conversation":
            validation_predicted[1][2] += 1
        if test_i == "publicité":
            validation_predicted[1][3] += 1
        if test_i == "bot":
            validation_predicted[1][4] += 1
        if test_i == "other spam":
            validation_predicted[1][5] += 1
    elif validation[0][i] == "conversation":
        validation_actual[2] += 1
        if test_i == "actualité":
            validation_predicted[2][0] += 1
        if test_i == "reaction":
            validation_predicted[2][1] += 1
        if test_i == "conversation":
            validation_predicted[2][2] += 1
        if test_i == "publicité":
            validation_predicted[2][3] += 1
        if test_i == "bot":
            validation_predicted[2][4] += 1
        if test_i == "other spam":
            validation_predicted[2][5] += 1
    elif validation[0][i] == "publicité":
        validation_actual[3] += 1
        if test_i == "actualité":
            validation_predicted[3][0] += 1
        if test_i == "reaction":
            validation_predicted[3][1] += 1
        if test_i == "conversation":
            validation_predicted[3][2] += 1
        if test_i == "publicité":
            validation_predicted[3][3] += 1
        if test_i == "bot":
            validation_predicted[3][4] += 1
        if test_i == "other spam":
            validation_predicted[3][5] += 1
    elif validation[0][i] == "bot":
        validation_actual[4] += 1
        if test_i == "actualité":
            validation_predicted[4][0] += 1
        if test_i == "reaction":
            validation_predicted[4][1] += 1
        if test_i == "conversation":
            validation_predicted[4][2] += 1
        if test_i == "publicité":
            validation_predicted[4][3] += 1
        if test_i == "bot":
            validation_predicted[4][4] += 1
        if test_i == "other spam":
            validation_predicted[4][5] += 1
    elif validation[0][i] == "other spam":
        validation_actual[5] += 1
        if test_i == "actualité":
            validation_predicted[5][0] += 1
        if test_i == "reaction":
            validation_predicted[5][1] += 1
        if test_i == "conversation":
            validation_predicted[5][2] += 1
        if test_i == "publicité":
            validation_predicted[5][3] += 1
        if test_i == "bot":
            validation_predicted[5][4] += 1
        if test_i == "other spam":
            validation_predicted[5][5] += 1

print(validation_predicted)
print(validation_actual)
