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
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import pickle
from sklearn import metrics
from sklearn.metrics import classification_report, f1_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

mbti = pd.read_csv("C:\\Users\\user\\Desktop\\kaggle_MBTI\\MBTI_prepro.csv")

#명사만 태깅
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

#원형화 위에거랑 합치면댐
def stemm(text):
    stemmer = SnowballStemmer(language='english')
    return stemmer.stem(text)

count_vect = CountVectorizer()


# Flag to re-create or not the machine learning model
recreate_model=False

# We'll save the model into a file:
filename = 'mbti_svm_v2.sav'

# If the model file doesn't exists
if not os.path.isfile(filename):
    recreate_model=True


X = mbti['posts'] # features
y = mbti['type']  # labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check if need to recreate the model
if recreate_model:    
    
    # Creating an instance to vectorizer:
    vectorizer = TfidfVectorizer()
    
    # Training the vectorizer:
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    # Training the classifier:
    clf = LinearSVC()
    clf.fit(X_train_tfidf, y_train)
    
    # Pipelining the vectorizer and the classifier
    text_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',LinearSVC())])
    text_clf.fit(X_train, y_train)
    
    # saving the model to disk
    pickle.dump(text_clf, open(filename, 'wb'))

# If there is no need to recreate the model, just open the file from the disk    
else:
    # loading the model from disk
    text_clf = pickle.load(open(filename, 'rb'))


predictions = text_clf.predict(X_test)
print(classification_report(y_test, predictions))




#이미 폴더에 명사만 있으니까 이거 counter vecterrizor 한다음 모델링 대신 4개로 나눠서 오케바리 ? -> 데이터는 준영이가 재수집하긴 해야댐
#아 그리고 구글 트랜스레이터도 가지고 오자 input 한국어를 영어로 치환
# 그리고 결과 바로 나오면 chatbot이랑 연동부터 해버리자

#mbti.to_csv("C:\\Users\\user\\Desktop\\kaggle_MBTI\\MBTI_prepro.csv", index = False)