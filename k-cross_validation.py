from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

# We take the classification algorithms that are used in classes.

df = pd.read_csv("NEW-ODI-FILTERED-STUDIES.CSV")
df.drop('Unnamed: 0', axis = 1, inplace = True)

target = 'What programme are you in?'
feature_columns = ['Have you taken a course on machine learning?', 
					'Have you taken a course on information retrieval?', 
					'Have you taken a course on statistics?', 
					'Have you taken a course on databases?']

X = df[feature_columns]
y = df[target]



NB = GaussianNB()

print '\nNaive Bayes:\n'

score = cross_val_score(NB, X, y, cv=10, scoring='accuracy').mean()

print '10-cross validation score: %.6f' % score



clf = tree.DecisionTreeClassifier()

print '\nDecision Tree:\n'

score = cross_val_score(clf, X, y, cv=10, scoring='accuracy').mean()

print '10-cross validation score: %.6f' % score

print '\nK-nearest neighbour classifier:\n'

k_range = list(range(1, 31))
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean()
    print '10-cross validation score with %d neighbours: %.6f' % (k, score)
