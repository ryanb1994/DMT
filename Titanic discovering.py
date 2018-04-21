import pandas as pd
import numpy as np
from statistics import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

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

# Family Members
df_train['Family members'] = df_train['SibSp'] + df_train['Parch']
df_test['Family members'] = df_test['SibSp'] + df_test['Parch']

# Alone yes or no
df_train['Alone'] = np.where(df_train['Family members'] < 1, 1, 0)
df_test['Alone'] = np.where(df_test['Family members'] < 1, 1, 0)

# Parent Yes or no
df_train['Parent'] = np.where((df_train['Parch'] > 0) & (df_train['Age'] > 25.0), 1, 0)
df_test['Parent'] = np.where((df_test['Parch'] > 0 ) & (df_test['Age'] > 25.0), 1, 0)

# Child Yes or no
df_train['Child'] = np.where((df_train['Parch'] > 0) & (df_train['Age'] <= 25) , 1, 0)
df_test['Child'] = np.where((df_test['Parch'] > 0) & (df_test['Age'] <= 25) , 1, 0)

# Fill NA embarked with most ocurrences embarked
df_train['Embarked'] = df_train['Embarked'].fillna(mode(df_train['Embarked']))
df_test['Embarked'] = df_test['Embarked'].fillna(mode(df_test['Embarked']))

# Fill NA Fare with the mean of all Fares
df_train['Fare'] = df_train.Fare.fillna(round(df_train['Fare'].median(), 4))
df_test['Fare'] = df_test.Fare.fillna(round(df_test['Fare'].median(), 4))

# Encode Embarked into digits
df_train['Embarked'] = encoder.fit_transform(df_train['Embarked'])
df_test['Embarked'] = encoder.fit_transform(df_test['Embarked'])

## Data dummies
df_train = pd.get_dummies(data=df_train, prefix=['Pclass'], columns=['Pclass'])
df_train = pd.get_dummies(data=df_train, prefix=['Embarked'], columns=['Embarked'])
# df_train = pd.get_dummies(data=df_train, prefix=['Family members'], columns=['Family members'])
df_test = pd.get_dummies(data=df_test, prefix=['Pclass'], columns=['Pclass'])
df_test = pd.get_dummies(data=df_test, prefix=['Embarked'], columns=['Embarked'])
# df_test = pd.get_dummies(data=df_test, prefix=['Family members'], columns=['Family members'])

# Drop Name and passengerId
df_survivors = df_train['Survived']
df_train = df_train.drop(['Name', 'PassengerId', 'Survived'], axis = 1)
df_testpss = df_test['PassengerId']
df_test = df_test.drop(['Name', 'PassengerId'], axis = 1)

df_train = df_train.drop(['Embarked_0', 'Pclass_3','Alone', 'Parent','Fare', 'Ticket', 'SibSp', 'Parch'], axis = 1)
df_test = df_test.drop(['Embarked_0', 'Pclass_3', 'Alone', 'Fare', 'Parent', 'Ticket','SibSp', 'Parch'], axis = 1)

# !!Combine all dataframes into one for last model ?

# print (df_train.head(40))
# print (df_test.head())
# print (df_test.dtypes)
# print (df_train.dtypes)


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

titanicdtsubmission1 = pd.DataFrame({
   "PassengerId": df_testpss,
   "Survived": ydt
})


titanicdtsubmission1.to_csv('Titanicanswer5.csv', index=False)


































