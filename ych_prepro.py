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


mbti = pd.read_csv("C:\\Users\\user\\Desktop\\kaggle_MBTI\\MBTI 500.csv")

mbti['E/I'] = mbti['type'].apply(lambda x: 0 if x[0] == 'E' else 1)
mbti['N/S'] = mbti['type'].apply(lambda x: 0 if x[1] == 'N' else 1)
mbti['F/T'] = mbti['type'].apply(lambda x: 0 if x[2] == 'F' else 1)
mbti['J/P'] = mbti['type'].apply(lambda x: 0 if x[3] == 'J' else 1)

mbti.to_csv("C:\\Users\\user\\Desktop\\kaggle_MBTI\\MBTI 500.csv")