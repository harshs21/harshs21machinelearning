#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression


# In[3]:


df = pd.read_csv("Downloads/ex1data1.csv", sep=",")


# In[4]:


df


# In[5]:


X = df.iloc[:, 0].values.reshape(-1, 1)
Y = df.iloc[:, 1].values.reshape(-1, 1)
linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)


# In[8]:


X.shape


# In[9]:


Y.shape


# In[10]:


plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.show()


# In[15]:




