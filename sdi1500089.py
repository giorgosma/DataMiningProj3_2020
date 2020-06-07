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
from nltk import word_tokenize
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

# region
# - ### *Delete all punctuation*
# url detection from https://www.w3resource.com/python-exercises/re/python-re-exercise-42.php
# /+a-z0-9 detection from https://stackoverflow.com/questions/37813152/replace-words-starting-with-a-backslash-in-python
# rest /+ and punctuation from https://stackoverflow.com/questions/21672514/replacing-punctuation-in-a-data-frame-based-on-punctuation-list
# endregion
# region
trainDF['Comment'] = trainDF['Comment'].str.replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',' ', regex=True)
#trainDF['Comment'] = trainDF['Comment'].str.replace("\\[a-z]|[\]", " ", regex=True)

trainDF['Comment'] = trainDF['Comment'].str.replace(r"\\+\w+", "", regex=True)

trainDF['Comment'] = trainDF['Comment'].str.replace("[^\w\s]", "", regex=True)

#trainDF['Comment'] = trainDF['Comment'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
#trainDF['Comment'] = trainDF['Comment'].apply(lambda x: re.split('http:\/\/.*', str(x))[0])

# trainDF['Comment'] = trainDF['Comment'].replace('https:.^\s\n\r','', regex=True)
# trainDF['Comment'] = trainDF['Comment'].replace('http:.^\s\n\r','', regex=True)
#trainDF["Comment"] = trainDF["Comment"].str.replace('[^\w\s]',' ')
#trainDF["Comment"] = trainDF["Comment"].str.replace('[^\w\s]','', regex=True)
#trainDF['Comment'] = trainDF['Comment'].str.replace('[/.,\/#!$%\^&\*;:=\-_`~/g]', ' ', regex=True)
#trainDF['Comment'] = trainDF['Comment'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
trainDF.to_csv('./data/deletedTrain.csv')

trainDF
# endregion

# region
testLabelsDf["Comment"] = testLabelsDf["Comment"].str.replace('[^\w\s]','')

testLabelsDf
# endregion

# region
testSetDf["Comment"] = testSetDf["Comment"].str.replace('[^\w\s]','')

testSetDf
# endregion


