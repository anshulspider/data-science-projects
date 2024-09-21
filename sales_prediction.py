#!/usr/bin/env python
# coding: utf-8

# importing the dependencies

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics


# In[4]:


#loading the dataset from csv file to the pandas dataframe
item_data = pd.read_csv('advertising.csv')


# In[5]:


#first five rows of the dataframe
item_data.head()


# In[6]:


item_data.shape


# In[7]:


#check for null values in the dataset
item_data.info()


# In[10]:


item_data.describe()


# data cleaning sir 

# In[11]:


# Checking Null values
item_data.isnull().sum()*100/item_data.shape[0]
# There are no NULL values in the dataset, hence it is clean


# In[13]:


# Outlier Analysis
fig, axs = plt.subplots(3, figsize = (5,5))
plt1 = sns.boxplot(item_data['TV'], ax = axs[0])
plt2 = sns.boxplot(item_data['Newspaper'], ax = axs[1])
plt3 = sns.boxplot(item_data['Radio'], ax = axs[2])
plt.tight_layout()


# no outliers in the rows of the given data

# Exploratory Data Analysis

# Sales (Target Variable)

# In[14]:


sns.boxplot(item_data['Sales'])
plt.show()


# In[15]:


# Let's see how Sales are related with other variables using scatter plot.
sns.pairplot(item_data, x_vars=['TV', 'Newspaper', 'Radio'], y_vars='Sales', height=4, aspect=1, kind='scatter')
plt.show()


# In[18]:


# Let's see the correlation between different variables.
sns.heatmap(item_data.corr(), cmap="YlGnBu", annot = True)
plt.show()


# As is visible from the pairplot and the heatmap, the variable TV seems to be most correlated with Sales. So let's go ahead and perform simple linear regression using TV as our feature variable.

# We first assign the feature variable, TV, in this case, to the variable X and the response variable, Sales, to the variable y.

# In[19]:


X = item_data['TV']
y = item_data['Sales']


# You now need to split our variable into training and testing sets. You'll perform this by importing train_test_split from the sklearn.model_selection library. It is usually a good practice to keep 70% of the data in your train dataset and the rest 30% in your test dataset

# In[20]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)


# # Let's now take a look at the train dataset
# 

# In[21]:


X_train.head()


# In[23]:


y_train.head()


# Building a Linear Model
# You first need to import the statsmodel.api library using which you'll perform the linear regression.

# In[24]:


import statsmodels.api as sm


# In[25]:


# Add a constant to get an intercept
X_train_sm = sm.add_constant(X_train)

# Fit the resgression line using 'OLS'
lr = sm.OLS(y_train, X_train_sm).fit()


# In[26]:


lr.params


# In[27]:


print(lr.summary())


# In[28]:


plt.scatter(X_train, y_train)
plt.plot(X_train, 6.948 + 0.054*X_train, 'r')
plt.show()


# In[29]:


y_train_pred = lr.predict(X_train_sm)
res = (y_train - y_train_pred)


# In[30]:


fig = plt.figure()
sns.distplot(res, bins = 15)
fig.suptitle('Error Terms', fontsize = 15)                  # Plot heading 
plt.xlabel('y_train - y_train_pred', fontsize = 15)         # X-label
plt.show()


# In[31]:


plt.scatter(X_train,res)
plt.show()


# In[32]:


# Add a constant to X_test
X_test_sm = sm.add_constant(X_test)

# Predict the y values corresponding to X_test_sm
y_pred = lr.predict(X_test_sm)


# In[33]:


y_pred.head()


# In[34]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[35]:


#Returns the mean squared error; we'll take a square root
np.sqrt(mean_squared_error(y_test, y_pred))


# In[36]:


r_squared = r2_score(y_test, y_pred)
r_squared


# In[37]:


plt.scatter(X_test, y_test)
plt.plot(X_test, 6.948 + 0.054 * X_test, 'r')
plt.show()


# In[ ]:




