#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
import scipy.io as scipy
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn import metrics
get_ipython().run_line_magic('matplotlib', 'inline')
import scipy.optimize as opt


# In[2]:


df = pd.read_csv("Downloads/ex2data2.csv", sep=',')


# In[36]:


plt.scatter(df.iloc[:,0], df.iloc[:,1])
plt.show()


# In[11]:


X = df.iloc[:, [0, 1]].values
y = df.iloc[:, 2].values
print(X, y)


# In[14]:


#plotting the data

#masking (for plotting two different variables)

mask=y==1
passed=plt.scatter(X[mask][0], X[mask][1])
failed=plt.scatter(X[~mask][0], X[~mask][1])
plt.ylim(-1, 1)
plt.xlim(-1, 1)
plt.xlabel('microchip test 1')
plt.ylabel('microchip test 2')
plt.legend((passed,failed),('passed','failed'))
plt.show()


# In[27]:


#mapfeaturing
#done so that a more accurate decision boundary can be made 
#2 features will not be sufficient as a linear boundary is not required here

def mapFeature(x1,x2):      
    degree=2                               
    out=np.ones(X.shape[0])[:,np.newaxis]
    for i in range(1,degree+1):
        for j in range(i+1):
            out=np.hstack((out,np.multiply(np.power(x1,i-j),np.power(x2,j))[:,np.newaxis]))          
    return out
x=mapFeature(df.iloc[:,0],df.iloc[:,1])


# In[28]:


x


# In[29]:


#setting the parameters
lamda=1
(m,n)=x.shape
theta=np.zeros((n,1))
y=y[:,np.newaxis]


# In[30]:


#defining sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))


# In[31]:


#defining Regularized Cost Function
def reg_costFunc(theta, x, y, lamda):
    m=len(y)
    j=(-1/m)*(y.T @ np.log(sigmoid(x@ theta)) + (1 - y.T) @ np.log(1 - sigmoid(x@ theta)))#this is different from earlier cost 
                                                                                          #besause in this case we have to 
                                                                                          #create the result in the form of an
                                                                                          #array so that it can be added to
                                                                                          #regularization term
    reg=(lamda/(2*m))*(theta[1:].T @ theta[1:])
    j=j+reg
    return j

j=reg_costFunc(theta, x, y, lamda)
print(j)


# In[32]:


#defining Gradient Descent Function
def lrGradientDescent(theta, x, y, labda):
    m = len(y)
    grad = np.zeros([m,1])
    grad = (1/m) * x.T @ (sigmoid(x @ theta) - y)
    grad[1:] = grad[1:] + (lamda / m) * theta[1:]
    return grad


# In[33]:


#Learning parameters using fmin_tnc
output = opt.fmin_tnc(func = reg_costFunc, x0 = theta.flatten(), fprime = lrGradientDescent,                          args = (x, y.flatten(), lamda))
theta = output[0]

print(theta)

print("\n")

j=reg_costFunc(theta, x, y, lamda)    
print(j)


# In[35]:


# making decision boundary

u = np.linspace(-1, 1.5, 50)
v = np.linspace(-1, 1.5, 50)
z = np.zeros((len(u), len(v)))

def mapFeatureForPlotting(x1, x2):
    degree = 2
    out = np.ones(1)
    for i in range(1, degree+1):
        for j in range(i+1):
            out = np.hstack((out, np.multiply(np.power(x1, i-j), np.power(x2, j))))
    return out

for i in range(len(u)):
    for j in range(len(v)):
        z[i,j]=np.dot(mapFeatureForPlotting(u[i],v[j]),theta)

mask = y.flatten() == 1
x = df.iloc[:,:-1]
passed = plt.scatter(X[mask][0], X[mask][1])
failed = plt.scatter(X[~mask][0], X[~mask][1])
plt.contour(u,v,z,0)
plt.xlabel('Microchip Test1')
plt.ylabel('Microchip Test2')
plt.legend((passed, failed), ('Passed', 'Failed'))
plt.show()


# In[ ]:




