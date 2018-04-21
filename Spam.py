# from sklearn.neighbors import *
# from sklearn.tree import *
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import HashingVectorizer
# from sklearn.calibration import *
# from sklearn.linear_model import *
# from sklearn.multiclass import *
# from sklearn.svm import *
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

cvt = CountVectorizer()
Xtraincounts = cvt.fit_transform(data['Message'])
df_words = pd.DataFrame(Xtraincounts.todense(), columns=cvt.get_feature_names())

df = pd.concat([data, df_words], axis=1)
print (df.shape)
df.drop([col for col,val in df.sum().iteritems() if val < 15], axis=1, inplace=True)
print (df.shape)
# df.to_csv('Spammessagesandinstances.csv')
# Check for the connection of spam
# Like if spam = 1 and column = 1 then connection = 1, and train classifiers on these connections
# This is apparently compute intensive so you could continue with this csv
