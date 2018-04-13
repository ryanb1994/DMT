# -*- coding: utf-8 -*-
"""
Created on Mon Apr 09 13:20:43 2018

@author: Maria
"""

import pandas as pd
import re
from collections import defaultdict
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("ODI-2018.csv")

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
print(df['Time you went to be Yesterday'])
for i in range(0,len(df)):
    if df['Time you went to be Yesterday'][i] != '' and float(df['Time you went to be Yesterday'][i]) > 23.59:
        print(i)
        print float(df['Time you went to be Yesterday'][i]) 

df['Time you went to be Yesterday'][85] = 23.3
df['Time you went to be Yesterday'][107] = 0
df['Time you went to be Yesterday'][129] = 23.59
df['Time you went to be Yesterday'][167] = 0
df['Time you went to be Yesterday'][211] = 3.3

likes_dict = defaultdict(int)
for i in range(0,len(df)):
    answer = (str(df['What makes a good day for you (1)?'][i]) + ' '+ \
              str(df['What makes a good day for you (2)?'][i])).lower()
    words = re.split('\W+', answer)
    for key in words:
        if key in likes_dict:
            likes_dict[key] += 1
        else:
            likes_dict[key] = 1

likes_dict ['sleep and derivatives'] = 0 
likes_dict ['food or food related'] = 0
likes_dict ['weather related'] = 0 
likes_dict ['others'] = 0
for key in likes_dict.keys():
    if key in ['good','well','nice','with','the','time','out','and','too', \
               'not','all', 'doing','down','for','have','that','going','lot', \
               'more', 'get','getting','some','having','day'] \
    or len(key) < 3:
        likes_dict.pop(key)    
    else:
        if ('sun' in key or 'weather' in key) and key != 'weather related':
            likes_dict ['weather related'] += likes_dict[key]
        else:
            if ('sleep' in key or 'slep' in key) and key != 'sleep and derivatives':
                likes_dict ['sleep and derivatives'] += likes_dict[key] 
            else:
                if  ('chocolate' in key or 'pizza' in key or 'dinner' in key or 'breakfast' in key or \
                'cheese' in key or 'sushi' in key or 'pancakes' in key or 'food' in key) and key != 'food or food related':
                    likes_dict ['food or food related'] += likes_dict[key]
                else:
                    if key not in ['sleep and derivatives', 'food or food related', 'weather related','coffee']:
                        likes_dict['others'] += likes_dict[key]
                       
for key, value in sorted(likes_dict.iteritems(), key=lambda (k,v): (v,k)):
    print "%s: %s" % (key, value)
    

    
objects = ['Others', 'Food', 'Weather','Sleep', 'Coffee']
y_pos = np.arange(len(objects))
nr_people = [475, 79, 75, 36, 22]
 
plt.bar(y_pos, nr_people, align='center', alpha=0.5, color = ['seagreen','firebrick', 'gold','gray','saddlebrown' ])
plt.xticks(y_pos, objects)
plt.ylabel('Number of answers')
plt.title('What makes people happy?')
plt.show()

sleep_late = 0
sleep_early = 0

for i in range(0,len(df)):
    time = df['Time you went to be Yesterday'][i]
    answer = (str(df['What makes a good day for you (1)?'][i]) + ' '+ \
              str(df['What makes a good day for you (2)?'][i])).lower()
    if 'sleep' in answer or 'slep' in answer or 'coff' in answer:
        if time != '' and float(time) >= 0 and float(time) <= 9:       
            sleep_late += 1
        else:
            sleep_early += 1
print(sleep_late,sleep_early) 
#more than 50% of the people who like either sleep or coffee went to bed late    
food_fat = 0
food_slim = 0
food_neither = 0
food_idk = 0

for i in range(0,len(df)):
    choc = df['Chocolate makes you.....'][i]
    key = (str(df['What makes a good day for you (1)?'][i]) + ' '+ \
              str(df['What makes a good day for you (2)?'][i])).lower()
    if  ('chocolate' in key or 'pizza' in key or 'dinner' in key or 'breakfast' in key or \
                'cheese' in key or 'sushi' in key or 'pancakes' in key or 'food' in key):
        if  'fat' in str(choc):           
                food_fat += 1
        if  'slim' in str(choc):
                food_slim += 1
        if 'neither' in str(choc):
            food_neither += 1
        if 'no idea' in str(choc):
            food_idk +=1
print(food_fat,food_slim,food_neither,food_idk)

values = [30,28]
keys = ['Went to bed late', 'Went to bed early']    
explode = [0.1,0]
plt.axis("equal")
plt.pie(values,labels=keys,explode=explode,autopct='%1.0f%%', shadow = 'true', startangle=90)
plt.title('People who like sleep or coffee')
plt.show()

values = [30, 7, 24, 12] 
keys = ['Fat','Slim','Neither',"I have no idea what you're talking about"]
explode = [0.1,0,0,0]
plt.axis("equal")
plt.pie(values,labels=keys,explode=explode,autopct='%1.0f%%', shadow = 'true', startangle=90)
plt.title('People who like food')
plt.show()

