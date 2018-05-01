import pandas as pd
import numpy as np
from statistics import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix  

encoder = LabelEncoder()

# Load in dataframes
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# Mapping dict for titles
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Other": 5}

## Data variables cleaning
# Drop Cabin
df_train = df_train.drop(['Cabin'], axis = 1)
df_test = df_test.drop(['Cabin'], axis = 1)

# Map Sex
df_train['Sex'] = df_train['Sex'].map({'male': 0, 'female': 1})
df_test['Sex'] = df_test['Sex'].map({'male': 0, 'female': 1})

# Remove non-digits from Ticket
df_train['Ticket'].replace(regex=True, inplace=True, to_replace=r'\D', value=r'')
df_test['Ticket'].replace(regex=True, inplace=True, to_replace=r'\D', value=r'')
# Maybe convert ticket to int
df_train['Ticket'] = np.where(df_train['Ticket'] == "", 0, df_train['Ticket'])
df_test['Ticket'] = np.where(df_test['Ticket'] == "", 0, df_test['Ticket'])
df_train['Ticket'] = df_train['Ticket'].astype(int)
df_test['Ticket'] = df_test['Ticket'].astype(int)

# Map Titles
df_train['Title'] = df_train.Name.str.extract(r' ([A-Za-z]+)\.', expand = False)
df_train['Title'] = df_train['Title'].map(titles)
df_train['Title'] = df_train['Title'].fillna(0)

df_test['Title'] = df_test.Name.str.extract(r' ([A-Za-z]+)\.', expand = False)
df_test['Title'] = df_test['Title'].map(titles)
df_test['Title'] = df_test['Title'].fillna(0)

# Age, Fill NAN with mean
df_train['Age'] = df_train.Age.fillna(round(df_train['Age'].median(), 2))
df_test['Age'] = df_test.Age.fillna(round(df_test['Age'].median(), 2))

# Age under 12
df_train['U12'] = np.where(df_train['Age'] < 12, 1, 0)
df_test['U12'] = np.where(df_test['Age'] < 12, 1, 0)
# Age above 12
df_train['A12'] = np.where(df_train['Age'] >= 12, 1, 0)
df_test['A12'] = np.where(df_test['Age'] >= 12, 1, 0)

# Family Members
df_train['Family members'] = df_train['SibSp'] + df_train['Parch']
df_test['Family members'] = df_test['SibSp'] + df_test['Parch']

# Fill NA embarked with most ocurrences embarked
df_train['Embarked'] = df_train['Embarked'].fillna(mode(df_train['Embarked']))
df_test['Embarked'] = df_test['Embarked'].fillna(mode(df_test['Embarked']))

# Encode Embarked into digits
df_train['Embarked'] = encoder.fit_transform(df_train['Embarked'])
df_test['Embarked'] = encoder.fit_transform(df_test['Embarked'])

# Drop Name and passengerId
df_survivors = df_train['Survived']
df_train = df_train.drop(['Name', 'PassengerId', 'Survived'], axis = 1)
df_testpss = df_test['PassengerId']
df_test = df_test.drop(['Name', 'PassengerId'], axis = 1)

df_train = df_train.drop(['Age', 'Fare', 'Ticket', 'SibSp', 'Parch'], axis = 1)
df_test = df_test.drop(['Age', 'Fare', 'Ticket','SibSp', 'Parch'], axis = 1)

# Logistic Regression
lr = LogisticRegression()
lr.fit(df_train, df_survivors)
y_pred = lr.predict(df_test)
acc_log = round(lr.score(df_train, df_survivors)*100, 2)
print (acc_log)

coeff_df = pd.DataFrame(df_train.columns)
coeff_df.columns = ['Feature']
coeff_df['Correlation'] = pd.Series(lr.coef_[0])
coeff_df.sort_values('Correlation', ascending=False)
print (coeff_df)

decision_tree = DecisionTreeClassifier()
decision_tree.fit(df_train, df_survivors)
ydt = decision_tree.predict(df_test)
acc_decision_tree = round(decision_tree.score(df_train, df_survivors) * 100, 2)
print (acc_decision_tree)

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(df_train, df_survivors)
yrf = random_forest.predict(df_test)
random_forest.score(df_train, df_survivors)
acc_random_forest = round(random_forest.score(df_train, df_survivors) * 100, 2)
print (acc_random_forest)


"""
# Export
titanicdtsubmission1 = pd.DataFrame({
   "PassengerId": df_testpss,
   "Survived": ydt
})
titanicdtsubmission1.to_csv('Titanictest.csv', index=False)
"""




















