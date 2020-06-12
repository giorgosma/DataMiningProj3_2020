# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_markers: region,endregion
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # <center>Data Mining Project 3 Spring semester 2019-2020</center>
# ## <center>Γεώργιος Μαραγκοζάκης &emsp; 1115201500089</center>

# ___

# ### Do all the necessary imports for this notebook

# region
# data processing
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords as nltkStopwords
from string import punctuation, digits
import re
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

# visualization
from wordcloud import WordCloud
from IPython.display import Image
from IPython.display import display
from itertools import cycle
import matplotlib.patches as mpatches

# classification
from sklearn.model_selection import KFold, cross_validate
from sklearn import svm, preprocessing
from sklearn.metrics import classification_report, make_scorer, accuracy_score, \
                            precision_score, recall_score, f1_score, roc_curve, auc,\
                            roc_auc_score, plot_roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
import scipy
from collections import Counter
import gensim 
import random
from operator import add



# vectorization
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# clustering
from nltk.cluster import KMeansClusterer, cosine_distance
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

# for data exploration
import os
import numpy as np
# endregion

# ## __Dataset Preprocessing__

# - ### *Read all .csv files*

specificPath = './data/'
# find the column's names of each csv
for fileName in os.listdir(specificPath):
    # we need to check only .csv files
    if fileName.endswith(".csv"):
        thisCsv = pd.read_csv(os.path.join(specificPath, fileName), dtype='unicode')
        if fileName == 'train.csv':
            trainDF = thisCsv
        elif fileName == 'impermium_verification_labels.csv':
            testLabelsDf = thisCsv
        elif fileName == 'impermium_verification_set.csv':
            testSetDf = thisCsv
        else:
            print("Wrong File!")

# __train.csv__

trainDF

# __test.csv__

testLabelsDf

testSetDf

# - ### *Convert all comments to lower case*

# region
trainDF['Comment'] = trainDF['Comment'].str.lower()

trainDF
# endregion

# region
testLabelsDf['Comment'] = testLabelsDf['Comment'].str.lower()

testLabelsDf
# endregion

# region
testSetDf['Comment'] = testSetDf['Comment'].str.lower()

testSetDf
# endregion

# - ### *Delete all punctuation, urls  and characters like “\n”, “\u0111”*

# Πάρθηκαν ιδέες και παραδείγματα από τα παρακάτω link:
# - url detection from https://www.w3resource.com/python-exercises/re/python-re-exercise-42.php
# - /+a-z0-9 detection from https://stackoverflow.com/questions/37813152/replace-words-starting-with-a-backslash-in-python
# - rest /+ and punctuation from https://stackoverflow.com/questions/21672514/replacing-punctuation-in-a-data-frame-based-on-punctuation-list

# region
trainDF['Comment'] = trainDF['Comment'].str.replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','', regex=True)

trainDF['Comment'] = trainDF['Comment'].str.replace(r'\\+\w+', '', regex=True)

trainDF['Comment'] = trainDF['Comment'].str.replace('[^\w\s]', '', regex=True)

trainDF
# endregion

# region
testLabelsDf['Comment'] = testLabelsDf['Comment'].str.replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','', regex=True)

testLabelsDf['Comment'] = testLabelsDf['Comment'].str.replace(r'\\+\w+', '', regex=True)

testLabelsDf['Comment'] = testLabelsDf['Comment'].str.replace('[^\w\s]', '', regex=True)

testLabelsDf
# endregion

# region
testSetDf['Comment'] = testSetDf['Comment'].str.replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','', regex=True)

testSetDf['Comment'] = testSetDf['Comment'].str.replace(r'\\+\w+', '', regex=True)

testSetDf['Comment'] = testSetDf['Comment'].str.replace('[^\w\s]', '', regex=True)

testSetDf
# endregion

# ## __Classification__

# - #### Classification using Naive Bayes classifier

def NaiveBayesClassification(trainX, trainY, testX, testY, labelEncoder):
    """
    Classify the text using the Naive Bayes classifier of scikit-learn    
    """

    clf = GaussianNB()
    
    trainX = trainX.toarray()
    
    # fit train set
    clf.fit(trainX, trainY)
    
    # Predict test set
    testX = testX.toarray()
    predY = clf.predict(testX)

    return accuracy_score(testY, predY), f1_score(testY, predY, average='weighted')

def NaiveBayesClassificationLS(trainX, trainY, testX, testY, labelEncoder):
    """
    Classify the text using the Naive Bayes classifier of scikit-learn for Laplace Smoothing  
    """

    clf = MultinomialNB(alpha=1.0)
    
    trainX = trainX.toarray()
    
    # fit train set
    clf.fit(trainX, trainY)
    
    # Predict test set
    testX = testX.toarray()
    predY = clf.predict(testX)

    return accuracy_score(testY, predY), f1_score(testY, predY, average='weighted')

# - #### Classification using SVM classifier

def SvmClassification(trainX, trainY, testX, testY, labelEncoder):
    """
    Classify the text using the SVM classifier of scikit-learn    
    """
    
    clf = svm.SVC(kernel='linear', C=1, probability=True)

    # fit train set
    clf.fit(trainX, trainY)
    
    # Predict test set
    predY = clf.predict(testX)

    return accuracy_score(testY, predY), f1_score(testY, predY, average='weighted')

# - #### Classification using Random Forests classifier

def RandomForestClassification(trainX, trainY, testX, testY, labelEncoder):
    """
    Classify the text using the Random Forest classifier of scikit-learn    
    """
    
    clf = RandomForestClassifier()

    # fit train set
    clf.fit(trainX, trainY)
    
    # Predict test set
    predY = clf.predict(testX)

    return accuracy_score(testY, predY), f1_score(testY, predY, average='weighted')

# ## __Vectorization__

# region
# build label encoder for Insults
le = preprocessing.LabelEncoder()
le.fit(trainDF["Insult"])

# transform Insults into numbers
trainY = le.transform(trainDF["Insult"])
testY = le.transform(testLabelsDf["Insult"])

accuracyF1Dict = dict()

print(type(le), " ", type(le.classes_))
print(le.classes_.dtype)
print(type(trainY))
print(trainY)
le.classes_
# endregion

trainY

testY


# - #### Bag-of-words vectorization

# region
# bowVectorizer = CountVectorizer()

# trainX = bowVectorizer.fit_transform(trainDF['Comment'])
# testX = bowVectorizer.transform(testSetDf['Comment'])

# # print('\n-------------Naive Bayes Classification with BOW Vectorization-------------')
# accuracyF1Dict["BOW-NB"] = NaiveBayesClassification(trainX, trainY, testX, testY, le)
# endregion

# - #### Adding Lemmatization

class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

# region
bowVectorizer = CountVectorizer(tokenizer=LemmaTokenizer())

trainX = bowVectorizer.fit_transform(trainDF['Comment'])
testX = bowVectorizer.transform(testSetDf['Comment'])

# print('\n-------------Naive Bayes Classification with BOW Vectorization and Lemmatization-------------')
accuracyF1Dict["BOW-NB-LM"] = NaiveBayesClassification(trainX, trainY, testX, testY, le)
# endregion

# - #### Removing Stop Words

# region
stopWords = ENGLISH_STOP_WORDS
stopWords = (stopWords.union(nltkStopwords.words('english')))

bowVectorizer = CountVectorizer(stop_words = stopWords)

trainX = bowVectorizer.fit_transform(trainDF['Comment'])
testX = bowVectorizer.transform(testSetDf['Comment'])

# print('\n-------------Naive Bayes Classification with BOW Vectorization and Lemmatization-------------')
accuracyF1Dict["BOW-NB-SW"] = NaiveBayesClassification(trainX, trainY, testX, testY, le)
# endregion

# - #### Adding Bigrams

# region
bowVectorizer = CountVectorizer(ngram_range = (2, 2))

trainX = bowVectorizer.fit_transform(trainDF['Comment'])
testX = bowVectorizer.transform(testSetDf['Comment'])

# print('\n-------------Naive Bayes Classification with BOW Vectorization and Lemmatization-------------')
accuracyF1Dict["BOW-NB-BG"] = NaiveBayesClassification(trainX, trainY, testX, testY, le)
# endregion

# - #### Adding Laplace Smoothing

# region
bowVectorizer = CountVectorizer()

trainX = bowVectorizer.fit_transform(trainDF['Comment'])
testX = bowVectorizer.transform(testSetDf['Comment'])

# print('\n-------------Naive Bayes Classification with BOW Vectorization and Lemmatization-------------')
accuracyF1Dict["BOW-NB-LS"] = NaiveBayesClassificationLS(trainX, trainY, testX, testY, le)
# endregion

# - #### Tf-idf vectorization

# region
tfIdfVectorizer = TfidfVectorizer()

trainX = tfIdfVectorizer.fit_transform(trainDF['Comment'])
testX = tfIdfVectorizer.transform(testSetDf['Comment'])

# print('-------------SVM Classification with TfIdf Vectorization-------------')
accuracyF1Dict["TfIdf-SVM"] = SvmClassification(trainX, trainY, testX, testY, le)

# print('\n-------------Random Forests Classification with TfIdf Vectorization-------------')
accuracyF1Dict["TfIdf-RandomForests"] = RandomForestClassification(trainX, trainY, testX, testY, le)

trainX
# endregion

# - #### Adding Part-of-Speech Based Features

# region
def countTextTag(TextTagList, tag):
    counter = 0
    for (x,y) in TextTagList:
        if y.startswith(tag):
            counter += 1
    return counter

# trainX = bowVectorizer.fit_transform(trainDF['Comment'])
# testX = bowVectorizer.transform(testSetDf['Comment'])

listOfComments = trainDF['Comment'].tolist()
partOfSpeechTagerList = [nltk.pos_tag(word_tokenize(comment)) for comment in listOfComments]
nn = []
vb = []
rb = []
jj = []
for item in partOfSpeechTagerList:
    # nn = collections.Counter([y for (x,y) in item if y.startswith("NN")])
    nn.append(countTextTag(item, 'NN') / len(item) if len(item) else 0)
    vb.append(countTextTag(item, 'VB') / len(item) if len(item) else 0)
    rb.append(countTextTag(item, 'RB') / len(item) if len(item) else 0)
    jj.append(countTextTag(item, 'JJ') / len(item) if len(item) else 0)

posDf = pd.DataFrame()
posDf['Noun'] = nn
posDf['Verb'] = vb
posDf['Adverb'] = rb
posDf['Adjective'] = jj

posDf
# print(partOfSpeechTagerList[1])
# print(nn)

# text = word_tokenize(trainDF.iloc[1]['Comment'])
# nltk.help.upenn_tagset('NN.*')
# nltk.help.upenn_tagset('VB.*')
# nltk.help.upenn_tagset('RB.*')
# nltk.help.upenn_tagset('JJ.*')
# nltk.pos_tag(text)

# endregion

# #### Results Summary

# region
# resultsData = {r'Naive Bayes': ['Baseline', 'Lemmatization', 'Stop Words', 'Bigrams', 'Laplace Smoothing'],  
#                'Accuracy': [accuracyF1Dict["BOW-NB"][0], accuracyF1Dict["BOW-NB-LM"][0], accuracyF1Dict["BOW-NB-SW"][0], accuracyF1Dict["BOW-NB-BG"][0], accuracyF1Dict["BOW-NB-LS"][0]],
#                'F1 score': [accuracyF1Dict["BOW-NB"][1], accuracyF1Dict["BOW-NB-LM"][1], accuracyF1Dict["BOW-NB-SW"][1], accuracyF1Dict["BOW-NB-BG"][1], accuracyF1Dict["BOW-NB-LS"][1]]}

# resultsData2 = {r'SVM': ['Baseline'],
#                 'Accuracy': [accuracyF1Dict["TfIdf-SVM"][0]],
#                 'F1 Score': [accuracyF1Dict["TfIdf-SVM"][1]]}

# resultsData3 = {r'RandomForests': ['Baseline'],
#                 'Accuracy': [accuracyF1Dict["TfIdf-RandomForests"][0]],
#                 'F1 Score': [accuracyF1Dict["TfIdf-RandomForests"][1]]}
# endregion

# region
# resultsDataFrame = pd.DataFrame(data=resultsData)
# resultsDataFrame
# endregion

# region
# resultsDataFrame = pd.DataFrame(data=resultsData2)
# resultsDataFrame
# endregion

# region
# resultsDataFrame = pd.DataFrame(data=resultsData3)
# resultsDataFrame
# endregion


