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
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import pickle
from googletrans import Translator
from nltk.corpus import stopwords
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

#한국어를 영어로 변환
def translate_word(word):
    translator = Translator()
    translated_word = translator.translate(word, src= 'ko')
    return translated_word.text

#챗봇으로 받은 데이터를 str으로 계속 쌓고 이를 명사, 원형화한 다음 모델이 인식할 수 있게 시리즈 형태로 변환?
def input_test(text):
    text = translate_word(text)
    text = pd.Series(text)
    text = text.apply(lambda x: x.lower())
    text = text.apply(extract_nouns)
    result = text.apply(stemm)
    return result


def model():
    recreate_model=False
    filename = 'C:\\Users\\user\\Desktop\\mbti_chatbot\\mbti_svm_v2.sav'
    
    if not os.path.isfile(filename):
        recreate_model=True

    X = mbti['posts'] # features
    y = mbti['type']  # labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if recreate_model:    
        vectorizer = TfidfVectorizer()
        X_train_tfidf = vectorizer.fit_transform(X_train)
        clf = LinearSVC()
        clf.fit(X_train_tfidf, y_train)
        text_clf = Pipeline([('tfidf',vectorizer),('clf',LinearSVC())])
        text_clf.fit(X_train, y_train)
        pickle.dump(text_clf, open(filename, 'wb'))
        return text_clf
    else:
        text_clf = pickle.load(open(filename, 'rb'))
        return text_clf
    

    #predictions = text_clf.predict(X_test)
    #print(classification_report(y_test, predictions))


'''
cnt = 2
text = ""
while cnt != 0:
    a = input("입력: ") 
    text += a
    cnt -= 1

text = translate_word(ext)
testset = input_test(text)
text_clf.predict(testset)
'''

if __name__=="__main__":
    text_clf = model()
    #a ="I really like being outside, I like being listened to and I like to set a time when I go on a trip, and I have a lot of imaginations, I love you"
    a = "난 밖에 있는게 좋구요 그리고 여행계획짜는걸 좋아해요, 사랑합니다"
    a = input_test(a)
    z = text_clf.predict(a)


#모델링 4개를 못했는데 어쩌지   -> 모델링은 어차피 이진분류모델 4개만 만들어버리면 댐
# 데이터를 ""에 계속 쌓는 느낌으로 가야댐! -> 나쁘지 않음 chatbot 연동대고 시작하면 댈듯

# 그리고 결과 바로 나오면 chatbot이랑 연동부터 해버리자
