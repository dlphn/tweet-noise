# Deep Learning Tweet Noise

[![Python](https://img.shields.io/badge/python-3.6-blue.svg?style=flat-square)](https://docs.python.org/3/)

> Spam detection on Twitter

## Description

Python school project developed by 3 students from CentraleSupélec : Hélène, Valentin, and Delphine.

We consider as spam anything that is not related to a news event or a reaction to one.

## Requirements

pandas
scikit-learn
tensorflow

## Project structure

- `CNN.py`: Text CNN model
- `RNN.py`: Text RNN model
- Binary Classification :
    - `data_helpersBinary.py` : contain a few ancillary functions
    - `dataPreparationBinary.py` : transform a csv with labellized tweets into 4 json : train_data_actualite, train_data_spam, test_data_actualite and test test_data_spam
    - `trainCNNBinary.py` : train a CNN model
    - `trainRNNBinary.py` : train a RNN model
    - `evalCNNBinary.py` : evaluate the CNN model
    - `evalRNNBinary.py` : evaluate the RNN model
- Multiclasses Classification :
    - `data_helpersMulti.py` : contain a few ancillary functions
    - `dataPreparationMulticlasses.py` : transform a csv with labellized tweets into 2 json : train_data_multi and test_data_multi
    - `trainCNNMulti.py` : train a CNN model
    - `trainRNNMulti.py` : train a RNN model
    - `evalCNNMulti.py` : evaluate the CNN model
    - `evalRNNMulti.py` : evaluate the RNN model
    - `labels.json` : json containing the names of the classes (made by trainRNNmMlti or train CNNMulti)

## Steps

#### 1. Data Acquisition

Fetch data from Twitter API.

#### 2. Data Preprocessing

For a binary classification :

Run `dataPreparationBinary.py` or `dataPreparationMulti.py` with a csv with the right format (tweets_all_2019-03-06.csv is provided as an example).

#### 3. Data Preparation & Model Training

For a binary classification :

Run `trainRNNBinary.py` or `trainCNNBinary.py`.

For a multiclasses classification :

Run `trainRNNMulti.py` or `trainCNNMulti.py`.

### 4. Model Evaluation

Run `eval***.py` to evaluate the model. To run it, modify model_dir.
