#!/usr/bin/env python
# coding: utf-8

# importing the dependencies

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


#loading dataset to pandas dataframe
credit_card_data = pd.read_csv('creditcard[1].csv')


# In[3]:


#first five rows of dataset
credit_card_data.head()


# In[4]:


credit_card_data.tail()


# In[5]:


#dataset information
credit_card_data.info()


# In[6]:


#checking missing values in each column
credit_card_data.isnull().sum()


# In[9]:


#distribution of legit transaction and fraudlent transaction
credit_card_data['Class'].value_counts()


# this dataset is highly unbalanced

# 0 --> normal transaction
# 
# 1 --> fraudulent transaction

# In[10]:


#sepetating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]


# In[11]:


print(legit.shape)
print(fraud.shape)


# In[12]:


#stastical measures of the data
legit.Amount.describe()


# In[13]:


fraud.Amount.describe()


# In[14]:


#compare the values for both transactions
credit_card_data.groupby('Class').mean()


# under sampling

# build a sample dataset containing simiilar distribution of normal trnasaction an d fraudlenetn transactions

# no. of fraudulent transactions ==>492

# In[15]:


legit_sample = legit.sample(n=492)


# conocatenating two dataframes

# In[17]:


new_dataset = pd.concat([legit_sample, fraud], axis = 0)


# In[18]:


new_dataset.head()


# In[19]:


new_dataset.tail()


# In[20]:


new_dataset['Class'].value_counts()


# In[22]:


new_dataset.groupby('Class').mean()


# splitting the data into features and targets

# In[23]:


X=new_dataset.drop(columns='Class' , axis = 1)
Y=new_dataset['Class']


# In[24]:


print(X)


# In[25]:


print(Y)


# split the data into trainig data and testing data

# In[33]:


X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y , random_state=2)


# In[34]:


print(X.shape,X_train.shape,X_test.shape)


# Model training

# Logistic Regression model

# In[35]:


model = LogisticRegression()


# In[36]:


#traiing the logicstic regression model with traiinign data
model.fit(X_train, Y_train)


# model evaluation 
# 
# 
# Accuracy score

# In[38]:


#accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction ,Y_train)


# In[39]:


print('Acccuracy on training data:', training_data_accuracy)


# In[41]:


# accurcy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction , Y_test)


# In[42]:


print('Accuracy score on Test data' , test_data_accuracy )


# In[ ]:




