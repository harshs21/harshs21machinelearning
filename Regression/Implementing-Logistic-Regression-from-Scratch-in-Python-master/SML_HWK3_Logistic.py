
# coding: utf-8

# In[1]:


#Importing packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
import scipy.io as scipy
from collections import defaultdict
get_ipython().magic('matplotlib inline')


# In[2]:


#Setting Working Directory
os.chdir("C:/Users/Sagar Ghiya/Desktop")


# In[3]:


#Loading data
data1 = scipy.loadmat('data1')


# In[4]:


type(data1)


# In[5]:


data1.keys()


# In[6]:


x_train = np.matrix(data1['X_trn'])
y_train = np.matrix(data1['Y_trn'])


# In[7]:


x_train.shape


# In[8]:


y_train.shape


# In[9]:


#Storing data as matrix 
x_test = np.matrix(data1['X_tst'])
y_test = np.matrix(data1['Y_tst'])


# In[10]:


x_test.shape


# In[11]:


y_test.shape


# In[12]:


#Data exploration for Train Data
plt.plot(x_train,y_train,'o')


# In[13]:


#Data Exploration for Test Data
plt.plot(x_test,y_test,'ro')


# In[14]:


# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))



# In[15]:


#Cost Function
def CostFunc(X, y, omega, lamb):
    y_hat = sigmoid(X * omega)
    one = np.multiply(y, np.log(y_hat))
    two = np.multiply((1 - y), np.log(1 - y_hat))
    reg = (lamb / 2 * len(X)) * np.sum(np.power(omega[1:,:], 2))
    return (-1/len(X)) * np.sum(one + two) + reg


# In[16]:


#Stochastic Gradient descent to obtain optimal omega
def sgd(X, y, theta, rho, minibatch_size, lamb=0, threshold=0.0001, iters=1000):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = theta.ravel().shape[1]
    cost = [np.inf]
    
    while True:
        for b in range(math.ceil(len(X)/minibatch_size)):
            # generates random samples without replacement, all unique
            random_idx = np.random.choice(len(X), size=min(len(X), minibatch_size), replace = False)

            # Get pair of (X, y) of the current minibatch/chunk
            X_mini = X[random_idx]
            y_mini = y[random_idx]
            
            error = sigmoid(np.dot(X_mini,theta)) - y_mini
            
            for j in range(parameters):
                term = np.multiply(error, X_mini[:,j])
                temp[j,0] = theta[j,0] - ((rho/ len(X_mini)) * (np.sum(term) + lamb * theta[j,0]))

            theta = temp
            
        cost.append(CostFunc(X, y, theta, lamb))
        
        #print(cost[-1])
        
        if (cost[-2]-cost[-1]) > 0 and (cost[-2]-cost[-1]) < threshold:
            break
        
    return theta, cost


# In[17]:


lambdas = [0.01,0.03,0.1,0.3,1]


# In[18]:


#Regularization function
def regularizationParameter(X, y, kfolds, alpha, minibatch_size, threshold=0.0001, lambda_list = lambdas):
    theta1 = np.matrix(np.zeros((X.shape[1],1)))
    
    shuffled_idx = np.random.choice(len(X), size=len(X), replace = False)
    idx_per_fold = math.floor(len(X)/kfolds)
    if len(X) % kfolds !=0:
        kfolds+=1
        
    lambda_errorlist = defaultdict(list)
    
    for lamb in lambdas:
        for k in range(kfolds):
            #holdout_idx = np.arange(k*idx_per_fold, (k+1)*idx_per_fold)
            holdout_idx = shuffled_idx[k*idx_per_fold : (k+1)*idx_per_fold]
            #X_holdout = X[k*idx_per_fold : (k+1)*idx_per_fold]
            #X_training = np.delete(X,holdout_idx, axis=0)
            # Holdout data - kth fold
            X_holdout = X[holdout_idx]
            y_holdout = y[holdout_idx]
            # Training data - other than kth fold
            X_training = X[~holdout_idx]
            y_training = y[~holdout_idx]
            
            omega1, _ = sgd(X_training, y_training, theta1, alpha, minibatch_size, lamb, threshold)
            holdout_error_lk = CostFunc(X_holdout, y_holdout, omega1, lamb)
            
            lambda_errorlist[lamb].append(holdout_error_lk)
            #print("{0}th holdout error(λ={1}): {2}".format(k, lamb, holdout_error_lk))
        print("Average error on k-holdout sets for lamda={0} is {1}".format(lamb,np.mean(lambda_errorlist[lamb])))
    lambda_optimal = min(lambda_errorlist, key= lambda l: np.mean(lambda_errorlist[l]))
    
    return lambda_optimal


# In[19]:


#For Plotting
def plot(X_train, y_train, X_test, y_test, theta, title):
    # Dataframe with X and y concatenated
    test_df = pd.DataFrame(data=np.column_stack((X_test[:,1:],y_test)), columns=['X1','X2','Y'])
    train_df = pd.DataFrame(data=np.column_stack((X_train[:,1:],y_train)), columns=['X1','X2','Y'])

    positive_test = test_df[test_df['Y'].isin([1])]  
    negative_test = test_df[test_df['Y'].isin([0])]

    positive_train = train_df[train_df['Y'].isin([1])]  
    negative_train = train_df[train_df['Y'].isin([0])]
    
    fig, ax = plt.subplots(figsize=(12,8))  

    w0 = theta.item(0)
    w1, w2 = theta.item(1), theta.item(2)

    x_hyperplane = np.arange(math.floor(train_df['X1'].min()), math.ceil(train_df['X1'].max()),0.1)
    y_hyperplane = (-w0-w1*x_hyperplane)/w2

    ax.scatter(positive_train['X1'], positive_train['X2'], s=50, marker='x', label='Train Positive', c='b')  
    ax.scatter(positive_test['X1'], positive_test['X2'], s=50, marker='x', label='Test Positive', facecolors='none', edgecolors='b')  
    
    ax.scatter(negative_train['X1'], negative_train['X2'], s=50, marker='o', label='Train Negative', c='r')  
    ax.scatter(negative_test['X1'], negative_test['X2'], s=50, marker='o', label='Test Negative', facecolors='none', edgecolors='r')  
    
    plt.plot(x_hyperplane, y_hyperplane, 'k-', label='Decision Boundary')

    plt.legend(bbox_to_anchor=(1, 0), loc="upper right", ncol=5,
                bbox_transform=fig.transFigure, fontsize = 'medium', columnspacing = 0.5)  
    
    ax.set_xlabel('X1')  
    ax.set_ylabel('X2') 
    plt.title(title)
    fig.tight_layout()
    plt.show()


# In[20]:


#Function to predict classes
def predict(theta, X):  
    prob = sigmoid(X * theta)
    return [1 if x >= 0.5 else 0 for x in prob]


# In[21]:


#Adding intercept to data 
X1_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
X1_test = np.hstack((np.ones((x_test.shape[0], 1)), x_test))


# In[22]:


#Finding regularization parameter lambda
alpha = 0.0001
minibatch_size = 5
kfolds_10 = 10

lambda_optimal1 = regularizationParameter(X1_train, y_train, kfolds_10, alpha, minibatch_size)
print("Optimal value for lamda:{0}".format(lambda_optimal1))


# In[23]:


#Assigning theta to 0
theta = np.matrix(np.zeros((X1_train.shape[1],1)))
theta


# In[24]:


#Training theta
theta_sg1, cost_sg1 = sgd(X1_train, y_train, theta,alpha, minibatch_size=10, lamb=lambda_optimal1, threshold=0.0001)
print("Theta for training data:\n",theta_sg1)


# In[25]:


#Classification accuracy on train data
predictions_trn = predict(theta_sg1, X1_train)  
correct = [1 if a == b[0] else 0 for (a, b) in zip(predictions_trn, y_train.tolist())]  
accuracy = (sum(map(int, correct)) / len(correct)*100)  
print('Classification Error = {0:.2f}'.format(correct.count(0)/len(correct)))
print('No of mis-classified Training data points: {0}'.format(correct.count(0)))
print('accuracy = {0:.2f}%'.format(accuracy))


# In[26]:


#Classification Accuracy on Test Data
predictions_tst = predict(theta_sg1, X1_test)  
correct = [1 if a == b[0] else 0 for (a, b) in zip(predictions_tst, y_test.tolist())]  
accuracy = (sum(map(int, correct)) / len(correct)*100)  
print('Classification Error = {0:.2f}'.format(correct.count(0)/len(correct)))
print('No of mis-classified Test data points: {0}'.format(correct.count(0)))
print('accuracy = {0:.2f}%'.format(accuracy))


# In[27]:


plot(X1_train, y_train, X1_test, y_test, theta_sg1, title = "Logistic Regression - Dataset 1")


# In[28]:


#DATASET 2


# In[29]:


data2 = scipy.loadmat('data2')


# In[30]:


X2_train, y2_train = np.matrix(data2['X_trn']), np.matrix(data2['Y_trn'])
X2_test, y2_test = np.matrix(data2['X_tst']), np.matrix(data2['Y_tst'])


# In[31]:


#Adding intercept
X2_train = np.hstack((np.ones((X2_train.shape[0], 1)), X2_train))
X2_test = np.hstack((np.ones((X2_test.shape[0], 1)), X2_test))


# In[32]:


#Calculating hyperparameter lambda
alpha = 0.0001
minibatch_size = 5
kfolds_10 = 10

lambda_optimal2 = regularizationParameter(X2_train, y2_train, kfolds_10, alpha, minibatch_size)
print("Optimal value for lamda:{0}".format(lambda_optimal2))


# In[33]:


theta2 = np.matrix(np.zeros((X2_train.shape[1],1)))
theta2


# In[34]:


#Training Theta
theta_sg2, cost_sg2 = sgd(X2_train, y2_train, theta2, alpha,minibatch_size=10, lamb=lambda_optimal2, threshold=0.0001)
print("Theta for training data(Stochastic Logistic Regression):\n",theta_sg2)


# In[35]:


#Classification Accuracy on Train Data
predictions_trn_2 = predict(theta_sg2, X2_train)  
correct = [1 if ((a == 1 and b[0] == 1) or (a == 0 and b[0] == 0)) else 0 for (a, b) in zip(predictions_trn_2, y2_train.tolist())]  
accuracy = (sum(map(int, correct)) / len(correct)*100)  
print('Classification Error = {0:.2f}'.format(correct.count(0)/len(correct)))
print('No of mis-classified Training data points: {0}'.format(correct.count(0)))
print('accuracy = {0:.2f}%'.format(accuracy))


# In[36]:


#Classification Accuracy on Test Data
predictions_tst_2 = predict(theta_sg2, X2_test)  
correct = [1 if ((a == 1 and b[0] == 1) or (a == 0 and b[0] == 0)) else 0 for (a, b) in zip(predictions_tst_2, y2_test.tolist())]  
accuracy = (sum(map(int, correct)) / len(correct)*100)  
print('Classification Error = {0:.2f}'.format(correct.count(0)/len(correct)))
print('No of mis-classified Test data points: {0}'.format(correct.count(0)))
print('accuracy = {0:.2f}%'.format(accuracy))


# In[37]:


plot(X2_train, y2_train, X2_test, y2_test, theta_sg2, title = "Logistic Regression - Dataset 2")


# In[38]:


#The classification error is more in part B as compared to part A. This is because dataset 1 has 2 classes which are well sperated from each other. So we get 
# a proper hyperplane which seperates both classes. However in dataset 2 classes are not well seperated. There are few points which are mixed and hence there is more error. 

