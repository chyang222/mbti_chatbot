#!/usr/bin/python
# -*- coding : utf-8 -*-
'''
 @author : ych
'''

''' import '''
import pandas as pd
import numpy as np
import logging
import os
import datetime
from logging.config import dictConfig
import re

from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download()
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

mbti = pd.read_csv("C:\\Users\\user\\Desktop\\kaggle_MBTI\\MBTI 500.csv")

def extract_nouns(text): 
    stop_words_list = stopwords.words('english')
    nouns = []
    result = []
    text = re.sub('[^a-zA-Z0-9]',' ',text).strip() 
    tokens = nltk.word_tokenize(text)
    
    for word in tokens:
        if word not in stop_words_list:
            result.append(word)

    tagged = nltk.pos_tag(result)
    for word, pos in tagged:
        if pos.startswith('NN'):  
            nouns.append(word)

    return ' '.join(nouns)

def count_vectorizer(vectorizer, df, train_mode):
    df.facts = df.facts.apply(extract_nouns)
    if train_mode:
        data = vectorizer.fit_transform(df.facts)
    else:
        data = vectorizer.transform(df.facts)
    
    X_party1 = vectorizer.transform(df['first_party'])
    X_party2 = vectorizer.transform(df['second_party'])

    data = np.concatenate([X_party1.todense(), X_party2.todense(), data.todense()], axis=1)

    return data

mbti.posts.apply(extract_nouns)