import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

#                          
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]
submission_df = pd.read_csv('gender_submission.csv')
g = sns.FacetGrid(train_df, col='Embarked')
#plt.show(g.map(plt.hist, 'Age', bins=20))
# grid = sns.FacetGrid(train_df, col='Embarked')
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
# grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
# plt.show(grid.map(plt.hist, 'Age', alpha=.5, bins=20))
# plt.show(grid.add_legend()

#  ,'Age','Fare','Embarked'

# print(test_df.head())

# print("***************************************************************************************************************")

# plt.show(grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep'))
train_df['Sex'] = train_df['Sex'].replace('male',0)
train_df['Sex'] = train_df['Sex'].replace('female',1)
train_df['Embarked'] = train_df['Embarked'].replace('C',0)
train_df['Embarked'] = train_df['Embarked'].replace('Q',1)
train_df['Embarked'] = train_df['Embarked'].replace('S',2)
train_df['Age'] = train_df['Age'].replace(np.NaN,train_df.describe()['Age']['mean'])


test_df['Sex'] = test_df['Sex'].replace('male',0)
test_df['Sex'] = test_df['Sex'].replace('female',1)
test_df['Embarked'] = test_df['Embarked'].replace('C',0)
test_df['Embarked'] = test_df['Embarked'].replace('Q',1)
test_df['Embarked'] = test_df['Embarked'].replace('S',2)
test_df['Age'] = test_df['Age'].replace(np.NaN,test_df.describe()['Age']['mean'])

test_df['Fare'] = test_df['Fare'].replace(np.NaN,test_df.describe()['Fare']['mean'])

train_df['Age'] = train_df['Age'].astype(np.int64)
train_df['Embarked'] = train_df['Embarked'].astype(np.float)
train_df['Fare'] = train_df['Fare'].astype(np.int64)

test_df['Age'] = test_df['Age'].astype(np.int64)
test_df['Embarked'] = test_df['Embarked'].astype(np.float)
test_df['Fare'] = test_df['Fare'].astype(np.int64)

# print(test_df.info())

train_df =  train_df[['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
x = train_df[['Pclass','Sex','SibSp','Parch','Age','Fare']].as_matrix()
y = train_df['Survived']
test= test_df[['Pclass','Sex','SibSp','Parch','Age','Fare']].as_matrix()
# ,'Embarked','Fare'
knn = KNeighborsClassifier(30)

knn.fit(x,y)
# print(knn.predict(test))
submission_df['Survived'] = pd.DataFrame(knn.predict(test))

submission_df.to_csv('submission.csv')