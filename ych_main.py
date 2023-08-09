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


mbti = pd.read_csv("C:\\Users\\user\\Desktop\\mbti_chatbot\\kaggle_MBTI\\MBTI 500.csv")

mbti['type_split'] = mbti['type'].apply(list)
