import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
%matplotlib inline

train = pd.read_csv("C:\\Users\\titanic\\train.csv")
test=pd.read_csv("C:\\Users\\titanic\\test.csv")

train.head()train.describe()

train.shape

train.info()

train.isnull().sum()

train=train.drop(columns='Cabin',axis=1)

train['Age'].fillna(train['Age'].mean(),inplace=True)

print(train['Embarked'].mode())

print(train['Embarked'].mode()[0])

train['Embarked'].fillna(train['Embarked'].mode()[0],inplace=True)

train.isnull().sum()

train.describe()

train['Survived'].value_counts()

sns.set()

sns.countplot('Survived',data=train)

train['Survived'].value_counts()

sns.countplot('Sex',data=train)

sns.countplot('Sex',hue='Survived',data=train)

sns.countplot('Pclass',data=train)

sns.countplot('Pclass',hue='Survived',data=train)

sns.countplot('Embarked',hue='Survived',data=train)

survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
women = train[train['Sex']=='female']
men = train[train['Sex']=='male']
ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)
ax.legend()
_ = ax.set_title('Male')

FacetGrid = sns.FacetGrid(train, row='Embarked', size=4.5, aspect=1.6)
FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )
FacetGrid.add_legend()

grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();

train['Sex'].value_counts()

train['Embarked'].value_counts()

train.replace({'Sex':{'male':0,'female':1},'Embarked':{'S':0,'C':1,'Q':2}},inplace=True)

train.head()

x=train.drop(columns=['PassengerId','Name','Ticket','Survived'],axis=1)
y=train['Survived']

print (x)

print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

print(x.shape,x_train.shape,x_test.shape)

model=LogisticRegression()

model.fit(x_train,y_train)

x_train_prediction=model.predict(x_train)
print(x_train_prediction)

training_data_accuracy=accuracy_score(y_train,x_train_prediction)
print('Accuracy score of training data : ',training_data_accuracy)

x_test_prediction=model.predict(x_test)
print(x_test_prediction)

testing_data_accuracy=accuracy_score(y_test,x_test_prediction)
print('Accuracy score of training data : ',testing_data_accuracy)

sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')

