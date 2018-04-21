from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.calibration import *
from sklearn.linear_model import *
from sklearn.multiclass import *
from sklearn.svm import *
import pandas as pd
import numpy as np
import re

# Get data in orderly manner
data = pd.read_csv('smscollection.csv', names=['Txt1', 'Txt2', 'Txt3'])
data = data.drop(0)
data['Spam'] = np.where(data.Txt1.str.contains("spam"), 1, 0)
data['Message'] = data[['Txt1', 'Txt2', 'Txt3']].astype(str).sum(axis=1)
data = data.drop(['Txt1', 'Txt2', 'Txt3'], axis=1)
data['Message'] = data.Message.str.replace('(ham;)', "")
data['Message'] = data.Message.str.replace('(spam;)', "")
# print(data.head())

dataspam = pd.DataFrame(np.where((data['Spam']== 1), data['Message']), 0)
cvt = CountVectorizer()
Xtraincounts = cvt.fit_transform(data['Message'])

df_words = pd.DataFrame(Xtraincounts.todense(), columns=cvt.get_feature_names())

# Iterate over columns df_words, check sum, and delete any column if sum is under 250?






# Create train and test set
print(data.head())
print(df_words.head())






























