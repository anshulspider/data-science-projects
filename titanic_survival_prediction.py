#!/usr/bin/env python
# coding: utf-8

# IMPORTNIG THE DEPENDENCIS

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[3]:


# load the data from csv file to pandas dataframe
titanic_data = pd.read_csv('tested[1].csv')


# In[4]:


# importing the first 5 rows of the dataset
titanic_data.head()


# In[5]:


# no. of rows and columns in dataset
titanic_data.shape


# In[6]:


# getting some information about the data
titanic_data.info()


# In[7]:


# checking the no. of missing values in each column 
titanic_data.isnull().sum()


# HANDLING THE MISSING VALUES

# In[8]:


# drop the "cabin" column completely from the dataset
titanic_data = titanic_data.drop(columns='Cabin', axis = 1)


# In[9]:


# replacing the missing values in 'age' column with mean value
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace = True)


# In[13]:


print(titanic_data['Fare'].mode())


# In[15]:


print(titanic_data['Fare'].mode()[0])


# In[17]:


titanic_data['Fare'].fillna(titanic_data['Fare'].mode()[0],inplace=True)


# In[18]:


titanic_data.isnull().sum()


# DATA ANALYSIS

# In[19]:


#getting some statstical measures about the data
titanic_data.describe()


# In[20]:


# finding the number of people survived and not survived
titanic_data['Survived'].value_counts()


# DATA VISUALIZATION

# In[21]:


sns.set()


# In[22]:


# making a count for "survived" column
sns.countplot(x='Survived', data = titanic_data)


# In[23]:


titanic_data['Sex'].value_counts()


# In[24]:


# making a count for "sex" column
sns.countplot(x='Sex', data = titanic_data)


# In[25]:


# no. of survivors gender wise 
sns.countplot(x='Sex' , hue= 'Survived' , data= titanic_data )


# In[26]:


sns.countplot(x='Pclass', hue = 'Survived', data=titanic_data)


# ENCODING THE CATEGORICAL COLUMN

# In[27]:


titanic_data['Sex'].value_counts()


# In[28]:


titanic_data['Embarked'].value_counts()


# In[29]:


#converting categorical columns 
titanic_data.replace({'Sex':{'male':0 , 'female':1}, 'Embarked':{'S':0 , 'C':1 , 'Q':2}} , inplace=True) 


# In[30]:


titanic_data.head()


# SEPERATING FEATURES AND TARGET

# In[31]:


X = titanic_data.drop(columns = ['PassengerId','Name','Ticket','Survived'], axis = 1)
Y = titanic_data['Survived']


# In[32]:


print(X)


# In[33]:


print(Y)


# SPLITTING THE DATA INTO TRAINING DATA AND TEST DATA

# In[35]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2 , random_state = 2)


# In[36]:


print(X.shape , X_train.shape,X_test.shape )


# MODEL TRAINING MODEL

# LOGISTIC REGRESSION

# In[37]:


model = LogisticRegression()


# In[38]:


# training the logistic regression model with training data
model.fit(X_train, Y_train)


# MODEL EVALUATION 

# ACCURACY SCORE

# In[41]:


# accuradcy on training data
X_train_prediction = model.predict(X_train)


# In[42]:


print(X_train_prediction)


# In[43]:


training_data_accuracy = accuracy_score(Y_train,X_train_prediction)
print('Accuracy score of training data:',training_data_accuracy)


# In[44]:


# accuracy on test data
X_test_prediction = model.predict(X_test)


# In[45]:


print(X_test_prediction)


# In[46]:


test_data_accuracy = accuracy_score(Y_train,X_train_prediction)
print('Accuracy score of test data:',test_data_accuracy)


# In[ ]:




