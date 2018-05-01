from sklearn.neighbors import *
from sklearn.tree import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.calibration import *
from sklearn.linear_model import *
from sklearn.metrics import accuracy_score
from sklearn.multiclass import *
from sklearn.model_selection import train_test_split
from sklearn.svm import *
import pandas as pd
import numpy as np
import re




data = pd.read_csv('Spammessagesandinstances.csv', encoding='utf-8')


data = data.drop('Message', axis=1)
data = data.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(data, data['Spam'], test_size=0.2)


X_train = X_train.drop('Spam', axis=1)
X_test = X_test.drop('Spam', axis=1)

# print (X_train.shape, y_train.shape)
# print (X_test.shape, y_test.shape)

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
ydt = decision_tree.predict(X_test)
print (accuracy_score(ydt,y_test))
