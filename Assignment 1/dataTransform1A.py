# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 17:06:53 2018

@author: Maria
"""

import pandas as pd
import re
import os
import dateutil.parser as dp


root = 'D:\Cursuri Master\DMT'

df = pd.read_csv(os.path.join(root,"ODI-2018.csv"))
for i in range(0,len(df)):
    dateb = str(df['When is your birthday (date)?'][i])
    try:
        df['When is your birthday (date)?'][i] = str(dp.parse(dateb))[:10]
    except ValueError:
        pass
    
    
non_decimal = re.compile(r'[^\d.]+')
for i in range(0,len(df)):
    df['Time you went to be Yesterday'][i] = str(df['Time you went to be Yesterday'][i]).replace(":",".")
    df['Time you went to be Yesterday'][i] = non_decimal.sub('',  str(df['Time you went to be Yesterday'][i]))
    if df['Time you went to be Yesterday'][i] == "12":
        df['Time you went to be Yesterday'][i] = "0"
    if str(df['Time you went to be Yesterday'][i]).endswith('00'):
        df['Time you went to be Yesterday'][i] = df['Time you went to be Yesterday'][i][:-2]
    if str(df['Time you went to be Yesterday'][i]).endswith('.'):
        df['Time you went to be Yesterday'][i] = df['Time you went to be Yesterday'][i][:-1]
    if str(df['Time you went to be Yesterday'][i]).endswith('.'):
        df['Time you went to be Yesterday'][i] = df['Time you went to be Yesterday'][i][:-1]
    if len(str(df['Time you went to be Yesterday'][i])) > 1 and \
    str(df['Time you went to be Yesterday'][i][0]) == "0" and \
    str(df['Time you went to be Yesterday'][i][1]) != '.':
        df['Time you went to be Yesterday'][i] = df['Time you went to be Yesterday'][i][1:]


df['Time you went to be Yesterday'][85] = 23.3
df['Time you went to be Yesterday'][107] = 0
df['Time you went to be Yesterday'][129] = 23.59
df['Time you went to be Yesterday'][167] = 0
df['Time you went to be Yesterday'][211] = 3.3

df.loc[df['Have you taken a course on machine learning?'] == 'yes', \
       'Have you taken a course on machine learning?']= 1
df.loc[df['Have you taken a course on machine learning?'] == 'no', \
        'Have you taken a course on machine learning?']= 0
df.loc[df['Have you taken a course on statistics?'] == 'sigma', \
       'Have you taken a course on statistics?'] = 0
df.loc[df['Have you taken a course on statistics?'] == 'mu', \
       'Have you taken a course on statistics?'] = 1
df.loc[df['Have you taken a course on databases?'] == 'ja', \
       'Have you taken a course on databases?'] = 1
df.loc[df['Have you taken a course on databases?'] == 'nee', \
       'Have you taken a course on databases?'] = 0


df.to_csv('processedData.csv', sep = ",", index = False)







