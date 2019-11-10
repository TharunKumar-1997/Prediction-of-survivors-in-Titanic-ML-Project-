# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:04:46 2019

@author: Sanjeev
"""
import pandas as pd
dataset = pd.read_csv(r'E:\ssn\machine learning\dataset\titanic\train.csv')

##to show the statistical properties
dataset.describe()
dataset.info()
##to show the number of null values
dataset.isnull().sum()

import matplotlib.pyplot as plt
import seaborn as sns

##to show the presence of null value in the dataset
sns.heatmap(dataset.isnull(),yticklabels=False,cbar=False,cmap='inferno')

sns.countplot(dataset.Survived)
sns.countplot(x='Survived',data=dataset)

##to drop the null valued rows in the dataset
dataset = dataset.dropna(subset=['Embarked'])

##to show the count of survivors based on sex
sns.countplot(x='Survived',hue='Sex',data=dataset)

##to show the distribution of data
sns.distplot(dataset.Age.dropna(),bins=40,kde=False)
sns.distplot(dataset.Age.dropna(),bins=40)

##to replace the null values in the dataset
sns.boxplot(x='Pclass',y='Age',data=dataset)
sns.boxplot(x='Sex',y='Age',data=dataset)

dataset.groupby('Age')['Pclass'].mean()
dataset.Age.value_counts()

dataset.Pclass.value_counts()
'''dataset.groupby('Pclass')['Age']=dataset.groupby('Pclass')['Age'].median()

dataset.groupby('Pclass')['Age'].fillna(dataset.groupby('Pclass')['Age'].median())'''

for i in dataset.Pclass:
    if (i==1):
        dataset.Age.fillna(37,inplace=True)
    elif (i==2):
        dataset.Age.fillna(29,inplace=True)
    else:
        dataset.Age.fillna(24,inplace=True)

dataset=dataset.drop(['Cabin','Name','Ticket'],axis=1)

dataset = dataset.drop(columns=['Cabin'])

dataset=pd.get_dummies(dataset,columns=['Sex','Embarked'],drop_first=True)

x=dataset.drop('Survived',axis=1)
y=dataset.Survived

import statsmodels.formula.api as sm
def vif_cal(input_data,dependent_col):
    x_var=input_data.drop([dependent_col],axis=1)
    xvar_names=x_var.columns
    for i in range(0,len(xvar_names)):
        y=x_var[xvar_names[i]]
        x=x_var[xvar_names.drop(xvar_names[i])]
        rsq=sm.ols("y~x",x_var).fit().rsquared
        vif=round(1/(1-rsq),2)
        print(xvar_names[i],"VIF:",vif)
        
vif_cal(dataset,'Survived')

dataset = dataset.drop(columns=['Parch','Embarked_Q'])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train,y_train)
##to find the class probabilities
y_pred=classifier.predict(x_test)
##to find the class labels
classifier.predict_proba(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

accuracy=(cm[0,0]+cm[1,1])/(cm[0,1]+cm[1,0]+cm[0,0]+cm[1,1])
print(accuracy)

# =============================================================================
# 
# =============================================================================
import pandas as pd
import numpy as np

dataset=pd.read_csv(r'E:\ssn\machine learning\dataset\titanic\train.csv')

dataset1=pd.get_dummies(dataset,columns=['Sex','Embarked'],drop_first=True)

x=dataset.drop('Survived',axis=1)
y=dataset.Survived

import statsmodels.formula.api as sm
regressor=sm.OLS(y,x).fit()
regressor.summary()

x1=x.iloc[:,[0,1,2,3,5,6,8]]
regressor1=sm.OLS(y,x1).fit()
regressor1.summary()
# =============================================================================
# 
# =============================================================================

test.info()
test.isnull().sum()

test = pd.read_csv(r'E:\ssn\machine learning\dataset\titanic\test.csv')

test = pd.get_dummies(test,columns=['Sex','Embarked'],drop_first=True)

test = test.drop(columns=['Parch','Embarked_Q'])

test = test.drop(['Cabin','Name','Ticket'],axis=1)

test.groupby('Pclass')['Age'].median()

test.Fare = test.Fare.fillna(test.Fare.median())

for i in test.Pclass:
    if (i==1):
        test.Age.fillna(42,inplace=True)
    elif (i==2):
        test.Age.fillna(26.5,inplace=True)
    else:
        test.Age.fillna(24,inplace=True)

x=dataset.drop('Survived',axis=1)
y=dataset.Survived

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x,y)
##to find the class probabilities
y_pred_1=classifier.predict(test)
##to find the class labels
classifier.predict_proba(test)

prediction = pd.DataFrame({'PassengerId':test.PassengerId,'Survived':y_pred_1})

prediction.to_csv('titanic_survived.csv',index=False)

sns.boxplot(test.Fare)






    




