from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

df = pd.read_csv("NEW-ODI-FILTERED-STUDIES.CSV")
df.drop('Unnamed: 0', axis = 1, inplace = True)

target = 'What programme are you in?'
feature_columns = ['Have you taken a course on machine learning?', 
					'Have you taken a course on information retrieval?', 
					'Have you taken a course on statistics?', 
					'Have you taken a course on databases?']

X = df[feature_columns]
y = df[target]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

neigh = KNeighborsClassifier(n_neighbors=15)
neigh.fit(X, y) 

print 'Decision tree\n'
print clf.predict(X)

print '\n15-Neighbors classifier\n'
print neigh.predict(X)
