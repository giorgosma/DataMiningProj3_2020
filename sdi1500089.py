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

# Δεύτερη συνάρτηση υλοποίησης Naive Bayes με χρήση MultinomialNB για το Laplace Smoothing.

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
# endregion

# Εφόσον τα Insults στο dataset μας είναι 0 ή 1 δεν χρειάζεται ο μετασχηματισμός τους μέσω του LabelEconder.

# Για λόγους portability του κώδικα κρατήθηκε ο μετασχηματισμός από τη 2η εργασία.

# - #### Bag-of-words vectorization

# region
bowVectorizer = CountVectorizer()

trainX = bowVectorizer.fit_transform(trainDF['Comment'])
testX = bowVectorizer.transform(testSetDf['Comment'])

accuracyF1Dict["BOW-NB"] = NaiveBayesClassification(trainX, trainY, testX, testY, le)
# endregion

# - #### Adding Lemmatization

# Η ιδέα πάρθηκε από https://scikit-learn.org/stable/modules/feature_extraction.html#customizing-the-vectorizer-classes

class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

# region
bowVectorizer = CountVectorizer(tokenizer=LemmaTokenizer())

trainX = bowVectorizer.fit_transform(trainDF['Comment'])
testX = bowVectorizer.transform(testSetDf['Comment'])

accuracyF1Dict["BOW-NB-LM"] = NaiveBayesClassification(trainX, trainY, testX, testY, le)
# endregion

# - #### Removing Stop Words

# region
stopWords = ENGLISH_STOP_WORDS
stopWords = (stopWords.union(nltkStopwords.words('english')))

bowVectorizer = CountVectorizer(stop_words = stopWords)

trainX = bowVectorizer.fit_transform(trainDF['Comment'])
testX = bowVectorizer.transform(testSetDf['Comment'])

accuracyF1Dict["BOW-NB-SW"] = NaiveBayesClassification(trainX, trainY, testX, testY, le)
# endregion

# - #### Adding Bigrams

# region
bowVectorizer = CountVectorizer(ngram_range = (2, 2))

trainX = bowVectorizer.fit_transform(trainDF['Comment'])
testX = bowVectorizer.transform(testSetDf['Comment'])

accuracyF1Dict["BOW-NB-BG"] = NaiveBayesClassification(trainX, trainY, testX, testY, le)
# endregion

# - #### Adding Laplace Smoothing

# region
bowVectorizer = CountVectorizer()

trainX = bowVectorizer.fit_transform(trainDF['Comment'])
testX = bowVectorizer.transform(testSetDf['Comment'])

accuracyF1Dict["BOW-NB-LS"] = NaiveBayesClassificationLS(trainX, trainY, testX, testY, le)
# endregion

# - #### Tf-idf vectorization

# region
tfIdfVectorizer = TfidfVectorizer()

trainX = tfIdfVectorizer.fit_transform(trainDF['Comment'])
testX = tfIdfVectorizer.transform(testSetDf['Comment'])

accuracyF1Dict["TfIdf-SVM"] = SvmClassification(trainX, trainY, testX, testY, le)

accuracyF1Dict["TfIdf-RandomForests"] = RandomForestClassification(trainX, trainY, testX, testY, le)

# endregion

# - #### Adding Part-of-Speech Based Features

# Συνάρτηση υπολογισμού πλήθους λέξεων με tag του τύπου:

# - Noun tag = NN.*
# - Verb tag = VB.*
# - Adverb tag = RB.*
# - Adjective tag = JJ.*

def countTextTag(TextTagList, tag):
    counter = 0
    for (x,y) in TextTagList:
        if y.startswith(tag):
            counter += 1
    return counter

# Συνάρτηση δημιουργίας λίστας λιστών των fractions των τεσσάρων κατηγοριών λέξεων για κάθε comment.

# Για τον tagger η ιδέα πάρθηκε από https://www.nltk.org/book/ch05.html 

# Για τη διαίρεση με το μηδέν η ιδέα πάρθηκε από https://stackoverflow.com/questions/27317517/make-division-by-zero-equal-to-zero

def createFractionCharacteristics(listOfComments):
    # print("start create")
    partOfSpeechTagerList = [nltk.pos_tag(word_tokenize(comment)) for comment in listOfComments]
    nn = []
    vb = []
    rb = []
    jj = []
    for item in partOfSpeechTagerList:
        nn.append(countTextTag(item, 'NN') / len(item) if len(item) else 0)
        vb.append(countTextTag(item, 'VB') / len(item) if len(item) else 0)
        rb.append(countTextTag(item, 'RB') / len(item) if len(item) else 0)
        jj.append(countTextTag(item, 'JJ') / len(item) if len(item) else 0)
    return [nn, vb, rb, jj]

# Συνάρτηση συνένωσης πίνακα tfIdf με τα 4 fractions που παρήχθησαν από το POS.

# Για τη συνένωση των δύο πινάκων η ιδέα πάρθηκε από https://numpy.org/doc/stable/reference/generated/numpy.hstack.html

def mergeTfIdfWithPOS(textX, fractionList):
    textX = textX.toarray()
    rows, columns = textX.shape
    textX2 = np.empty([rows, columns + 4])
    for i in range(len(textX)):
        textX2[i] = np.hstack((textX[i], np.array((fractionList[0][i], fractionList[1][i], fractionList[2][i], fractionList[3][i]))))
    return textX2

# region
trainX2 = mergeTfIdfWithPOS(trainX, createFractionCharacteristics(trainDF['Comment'].tolist()))

testX2 = mergeTfIdfWithPOS(testX, createFractionCharacteristics(testSetDf['Comment'].tolist()))

accuracyF1Dict["TfIdf-POS-SVM"] = SvmClassification(trainX2, trainY, testX2, testY, le)

accuracyF1Dict["TfIdf-POS-RandomForests"] = RandomForestClassification(trainX2, trainY, testX2, testY, le)
# endregion

# ## __Beat the Benchmark__

# Lets add Lemmatization to TfIdf and run it with SVM 

# Από όλους τους συνδυασμούς που έκανα αυτός ήταν ο καλύτερος που οριακά έφτανε το καλύτερο μου score.

# region
bowVectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer())

trainX = bowVectorizer.fit_transform(trainDF['Comment'])
testX = bowVectorizer.transform(testSetDf['Comment'])

accuracyF1Dict["TfIdf-SVM-LM"] = SvmClassification(trainX, trainY, testX, testY, le)
# endregion

# #### Results Summary

# region
resultsData = {r'Naive Bayes': ['Baseline', 'Lemmatization', 'Stop Words', 'Bigrams', 'Laplace Smoothing'],  
               'Accuracy': [accuracyF1Dict["BOW-NB"][0], accuracyF1Dict["BOW-NB-LM"][0], accuracyF1Dict["BOW-NB-SW"][0], accuracyF1Dict["BOW-NB-BG"][0], accuracyF1Dict["BOW-NB-LS"][0]],
               'F1 score': [accuracyF1Dict["BOW-NB"][1], accuracyF1Dict["BOW-NB-LM"][1], accuracyF1Dict["BOW-NB-SW"][1], accuracyF1Dict["BOW-NB-BG"][1], accuracyF1Dict["BOW-NB-LS"][1]]}

resultsData2 = {r'SVM': ['Baseline', 'TfIdf + POS'],
                'Accuracy': [accuracyF1Dict["TfIdf-SVM"][0], accuracyF1Dict["TfIdf-POS-SVM"][0]],
                'F1 Score': [accuracyF1Dict["TfIdf-SVM"][1], accuracyF1Dict["TfIdf-POS-SVM"][1]]}

resultsData3 = {r'RandomForests': ['Baseline', 'TfIdf + POS'],
                'Accuracy': [accuracyF1Dict["TfIdf-RandomForests"][0], accuracyF1Dict["TfIdf-POS-RandomForests"][0]],
                'F1 Score': [accuracyF1Dict["TfIdf-RandomForests"][1], accuracyF1Dict["TfIdf-POS-RandomForests"][1]]}

resultsData4 = {r'Beat the Benchmark': ['TfIdf + SVM + LM'],
                'Accuracy': [accuracyF1Dict["TfIdf-SVM-LM"][0]],
                'F1 Score': [accuracyF1Dict["TfIdf-SVM-LM"][1]]}
# endregion

resultsDataFrame = pd.DataFrame(data=resultsData)
resultsDataFrame

resultsDataFrame = pd.DataFrame(data=resultsData2)
resultsDataFrame

resultsDataFrame = pd.DataFrame(data=resultsData3)
resultsDataFrame

resultsDataFrame = pd.DataFrame(data=resultsData4)
resultsDataFrame

# **Συμπεράσματα**
# - To dataset περιέχει πολλές λέξεις οι οποίες εκφράζουν επιθετικότητα αλλά ορισμένα γράμματα τους έχουν αντικατασταθεί από ειδικούς χαρακτήρες όπως @ και άλλοι. Επιπλέον με τον καθαρισμό του αρχείου από τους ειδικούς χαρακτήρες αυτές οι λέξεις δεν μπορούν να αναγνωριστούν πλέον.
# - Η υλοποίηση των αλγορίθμων ταξινόμισης βασίζονται στην 2η εργασία.
# - Η εκτέλεση των αλγορίμων SVM και RandomForest με TfIdf + Part-of-Speech χρειάζονται γύρω στα 15-17 λεπτά στο μηχάνημα μου, και αυτό προκύπτει από το μεγάλο μέγεθος των παραγόμενων πινάκων.
# - Στο Part-of-Speech για κάθε μια από τις 4 κατηγορίες χαρακτηριστικών λαμβάνονται υπόψιν και παράγωγα όπως για verb είναι τα VB, VBZ και άλλα.



