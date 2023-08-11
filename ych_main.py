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
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

mbti = pd.read_csv("C:\\Users\\user\\Desktop\\kaggle_MBTI\\MBTI_prepro.csv")
mbti.drop("Unnamed: 0", axis=1, inplace=True)

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

mbti.posts = mbti.posts.apply(extract_nouns)


#이미 폴더에 명사만 있으니까 이거 counter vecterrizor 한다음 모델링 대신 4개로 나눠서 오케바리 ? -> 데이터는 준영이가 재수집하긴 해야댐
#아 그리고 구글 트랜스레이터도 가지고 오자 input 한국어를 영어로 치환
# 그리고 결과 바로 나오면 chatbot이랑 연동부터 해버리자

#mbti.to_csv("C:\\Users\\user\\Desktop\\kaggle_MBTI\\MBTI_prepro.csv", index = False)