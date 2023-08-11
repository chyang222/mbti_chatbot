#!/usr/bin/python
# -*- coding : utf-8 -*-
'''
 @author : ych
'''

''' import '''
import pandas as pd
import numpy as np
from logging.config import dictConfig
import plotly.offline as pyo
import plotly.graph_objs as go


mbti = pd.read_csv("C:\\Users\\user\\Desktop\\kaggle_MBTI\\MBTI 500.csv")

mbti.head()
mbti['posts'][0]
mbti['type'].unique()
mbti.isnull().sum()


df_bar_chart=mbti.groupby('type').count()
trace1 = go.Bar(x=df_bar_chart.index, y=df_bar_chart['posts'])
data = [trace1]
layout = go.Layout(title='MBTI # Classified Posts per Type')
fig = go.Figure(data=data, layout=layout)
fig.show()




mbti['E/I'] = mbti['type'].apply(lambda x: 0 if x[0] == 'E' else 1)
mbti['N/S'] = mbti['type'].apply(lambda x: 0 if x[1] == 'N' else 1)
mbti['F/T'] = mbti['type'].apply(lambda x: 0 if x[2] == 'F' else 1)
mbti['J/P'] = mbti['type'].apply(lambda x: 0 if x[3] == 'J' else 1)

mbti.to_csv("C:\\Users\\user\\Desktop\\kaggle_MBTI\\MBTI 500.csv", index = False)