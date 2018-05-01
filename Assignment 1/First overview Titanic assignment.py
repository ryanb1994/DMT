# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 21:17:29 2018

@author: Jeffrey
"""

import pandas as pd
import numpy as np
from statistics import mode
import time as t

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

encoder = LabelEncoder()

#Dataframes
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df_train = df_train.drop(['Ticket', 'Cabin'], axis = 1)
df_test = df_test.drop(['Ticket', 'Cabin'], axis = 1)
df = [df_train, df_test]

titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}

for dat in df:
    dat['Title'] = dat.Name.str.extract(r' ([A-Za-z]+)\.', expand = False)
    dat['Title'] = dat['Title'].map(titles)
    dat['Title'] = dat['Title'].fillna(0)
    dat['Sex'] = dat['Sex'].map( {'male': 1, 'female': 0} ).astype(int)
    dat['Age'] = dat.Age.fillna(round(dat['Age'].mean(), 2))
    dat['Family members'] = dat['SibSp'] + dat['Parch']
    dat['Alone'] = np.where(dat['Family members'] < 1, 1, 0)
    dat['Embarked'] = dat['Embarked'].fillna(mode(dat['Embarked']))
    dat['Fare'] = dat.Fare.fillna(round(dat['Fare'].mean(), 4))
    dat['Embarked'] = encoder.fit_transform(dat['Embarked'])
    dat['Title'] = dat['Title'].astype(int)

df_train = pd.get_dummies(data=df_train, prefix=['Pclass'], columns=['Pclass'])
df_train = pd.get_dummies(data=df_train, prefix=['Embarked'], columns=['Embarked'])
df_test = pd.get_dummies(data=df_test, prefix=['Pclass'], columns=['Pclass'])
df_test = pd.get_dummies(data=df_test, prefix=['Embarked'], columns=['Embarked'])
df_train = df_train.drop(['Name', 'PassengerId', 'Embarked_0', 'Pclass_1'], axis = 1)
df_test = df_test.drop(['Name', 'Embarked_0', 'Pclass_1'], axis = 1)

df_trainset = df_train.drop('Survived', axis=1)
df_trainsets = df_train['Survived']
df_tests = df_test.drop("PassengerId", axis=1).copy()

st_time = t.clock()
# Logistic Regression
Logreg = LogisticRegression()
Logreg.fit(df_trainset, df_trainsets)
y_pred = Logreg.predict(df_tests)
acc_log = round(Logreg.score(df_trainset, df_trainsets)*100, 2)
print acc_log, "--- %s seconds ---" % (t.clock() - st_time)

st_time = t.clock()
coeff_df = pd.DataFrame(df_train.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df['Correlation'] = pd.Series(Logreg.coef_[0])
coeff_df.sort_values(by='Correlation', ascending=False)
print coeff_df, "--- %s seconds ---" % (t.clock() - st_time)

st_time = t.clock()
# Support Vector Machines
svc = SVC()
svc.fit(df_trainset, df_trainsets)
ysvc = svc.predict(df_tests)
acc_svc = round(svc.score(df_trainset, df_trainsets)*100, 2)
print acc_svc, "SVC", "--- %s seconds ---" % (t.clock() - st_time)

st_time = t.clock()
# KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(df_trainset, df_trainsets)
yknn = knn.predict(df_trainset)
acc_knn = round(knn.score(df_trainset, df_trainsets) * 100, 2)
print acc_knn, "KNN", "--- %s seconds ---" % (t.clock() - st_time)

st_time = t.clock()
#Gaussian NB
gaussian = GaussianNB()
gaussian.fit(df_trainset, df_trainsets)
ygssn = gaussian.predict(df_trainset)
acc_gaussian = round(gaussian.score(df_trainset, df_trainsets) * 100, 2)
print acc_gaussian, "Gaussian", "--- %s seconds ---" % (t.clock() - st_time)

st_time = t.clock()
# Perceptron
perceptron = Perceptron()
perceptron.fit(df_trainset, df_trainsets)
Yprep = perceptron.predict(df_trainset)
acc_perceptron = round(perceptron.score(df_trainset, df_trainsets) * 100, 2)
print acc_perceptron, "Perceptron", "--- %s seconds ---" % (t.clock() - st_time)

st_time = t.clock()
# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(df_trainset, df_trainsets)
ylsvc = linear_svc.predict(df_trainset)
acc_linear_svc = round(linear_svc.score(df_trainset, df_trainsets) * 100, 2)
print acc_linear_svc, "linear SVC", "--- %s seconds ---" % (t.clock() - st_time)

st_time = t.clock()
# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(df_trainset, df_trainsets)
ysgd = sgd.predict(df_trainset)
acc_sgd = round(sgd.score(df_trainset, df_trainsets) * 100, 2)
print acc_sgd, "sgd", "--- %s seconds ---" % (t.clock() - st_time)

st_time = t.clock()
# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(df_trainset, df_trainsets)
ydt = decision_tree.predict(df_trainset)
acc_decision_tree = round(decision_tree.score(df_trainset, df_trainsets) * 100, 2)
print acc_decision_tree, 'DT', "--- %s seconds ---" % (t.clock() - st_time)

st_time = t.clock()
# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(df_trainset, df_trainsets)
yrf = random_forest.predict(df_trainset)
random_forest.score(df_trainset, df_trainsets)
acc_random_forest = round(random_forest.score(df_trainset, df_trainsets) * 100, 2)
print acc_random_forest, "rf", "--- %s seconds ---" % (t.clock() - st_time)

