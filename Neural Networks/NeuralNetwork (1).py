#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

def relu(Z):
    R = np.maximum(0, Z)
    return R

def sigmoid(Z):
    S = 1 / (1 + np.exp(-1 * Z))
    return S

def relu_derivative(Z):
    Z[Z >= 0] = 1
    Z[Z < 0]  = 0
    return Z

def sigmoid_derivative(Z):
    SD = sigmoid(Z) * (1 - sigmoid(Z))
    return SD


# In[2]:


class model():
    
    def __init__(self):
        
        self.layers = []
        self.L = 0
        self.W = {}
        self.b = {}
        self.A = {}
        self.Z = {}
        self.dA = {}
        self.dZ = {}
        self.dW = {}
        self.db = {}
        self.cost = 0.
        self.m = 0
        self.lam = 0
        self.cost_history = []
        self.acc_history = []
        self.alpha_history = []
        self.alpha = 0.
        self.iterations = 0

        return
    
    def add_layers(self, list_of_layers):
        
        self.layers = list_of_layers
        self.L = len(self.layers) - 1 # Number of layers excluding the input feature layer
        
        return
    
    def init_params(self):
        
        for i in range(1, self.L + 1):
            self.W[str(i)] = np.random.randn(self.layers[i], self.layers[i - 1]) * np.sqrt(2. / self.layers[i - 1])
            self.b[str(i)] = np.zeros((self.layers[i], 1))
            
        return
    
    def forward_prop(self, X):
        
        self.A['0'] = X
        
        for i in range(1, self.L + 1):
            self.Z[str(i)] = np.dot(self.W[str(i)], self.A[str(i - 1)]) + self.b[str(i)]
            if i == self.L:
                # Output layer, Sigmoid activation
                self.A[str(i)] = sigmoid(self.Z[str(i)])
            else:
                # Hidden layer, Relu activataion
                self.A[str(i)] = relu(self.Z[str(i)])
        
        return
    
    def compute_cost(self, Y):
        
        self.cost = -1 * np.sum(np.multiply(Y, np.log(self.A[str(self.L)])) + 
                           np.multiply(1 - Y, np.log(1 - self.A[str(self.L)]))) / self.m 
        
        if self.lam != 0:
            reg = (self.lam / (2 * self.m))
            for i in range(1, self.L + 1):
                reg += np.sum(np.dot(self.W[str(i)], self.W[str(i)].T))
            self.cost += reg
            
        self.cost_history.append(self.cost)
        
        return
    
    def backward_prop(self, Y):
        
        # We need dA[str(L)] to start the backward prop computation
        self.dA[str(self.L)] = -1 * (np.divide(Y, self.A[str(self.L)]) - np.divide(1 - Y, 1 - self.A[str(self.L)]))
        self.dZ[str(self.L)] = np.multiply(self.dA[str(self.L)], sigmoid_derivative(self.Z[str(self.L)]))
        self.dW[str(self.L)] = np.dot(self.dZ[str(self.L)], self.A[str(self.L - 1)].T) / self.m + (self.lam/self.m) * self.W[str(self.L)]
        self.db[str(self.L)] = np.sum(self.dZ[str(self.L)], axis = 1, keepdims = True) / self.m
        self.dA[str(self.L - 1)] = np.dot(self.W[str(self.L)].T, self.dZ[str(self.L)])
            
        for i in reversed(range(1, self.L)):

            self.dZ[str(i)] = np.multiply(self.dA[str(i)], relu_derivative(self.Z[str(i)]))
            self.dW[str(i)] = np.dot(self.dZ[str(i)], self.A[str(i - 1)].T) / self.m + (self.lam/self.m) * self.W[str(i)]
            self.db[str(i)] = np.sum(self.dZ[str(i)], axis = 1, keepdims = True) / self.m
            self.dA[str(i - 1)] = np.dot(self.W[str(i)].T, self.dZ[str(i)])
        
        return
    
    def update_params(self):
        
        for i in range(1, self.L + 1):
            self.W[str(i)] = self.W[str(i)] - self.alpha * self.dW[str(i)]
            self.b[str(i)] = self.b[str(i)] - self.alpha * self.db[str(i)]
        
        return
    
    def train(self, X, Y, iterations = 10, 
        alpha = 0.001, decay = True, decay_iter = 5, decay_rate = 0.9, stop_decay_counter = 100,
        verbose = True, lam = 0):
        
        self.m = Y.shape[1]
        self.alpha = alpha
        self.iterations = iterations
        self.lam = lam
        
        # initialize parameters
        self.init_params()

        for i in range(iterations):
            # forward prop
            self.forward_prop(X)
            # compute cost
            self.compute_cost(Y)
            # backward prop
            self.backward_prop(Y)
            # update params
            self.update_params()
            # evaluate
            self.acc_history.append(self.evaluate(X, Y, in_training = True))
            # save alpha
            self.alpha_history.append(self.alpha)
            # learning rate decay
            if decay and stop_decay_counter > 0 and i % decay_iter == 0:
                self.alpha = decay_rate * self.alpha
                stop_decay_counter -= 1
            # display cost per iteration
            if verbose:
                print('Cost after {} iterations: {}'.format(i, self.cost))
        
        return
    
    def predict(self, X, in_training = False):
        
        if in_training == False:
            self.forward_prop(X)
            
        preds = self.A[str(self.L)] >= 0.5
        preds = np.squeeze(preds)
        
        return preds
        
    def evaluate(self, X, Y, in_training = False):
        
        examples = X.shape[1]
        
        pred = self.predict(X, in_training = in_training)
        pred = pred.reshape(1, examples)
        diff = np.sum(abs(pred - Y))
        acc = (examples - np.sum(diff)) / examples
        
        return acc
    
    def plot_cost(self):
        
        plt.plot(range(self.iterations), self.cost_history, color = 'b')
        plt.show()
        
        return
    
    def plot_acc(self):
        
        plt.plot(range(self.iterations), self.acc_history, color = 'r')
        plt.show()
        
        return
    
    def plot_alpha(self):
        
        plt.plot(range(self.iterations), self.alpha_history, color = 'g')
        plt.show()
        
        return


# In[3]:


m = model()
m.add_layers([2, 2, 1])
X = np.array([
    [1., 0., 0., 0., -10., 9.],
    [0., 1., 0., -6., -4., -1.]
])
Y = np.array([1, 1, 1, 0, 0, 1])
Y = Y.reshape(1, 6)

m.train(X, Y, iterations = 50, alpha = 0.1, verbose = False)


# In[4]:


m.plot_cost()
m.plot_acc()


# In[5]:


print(m.evaluate(X, Y))


# In[6]:


print(m.predict(X))


# In[9]:


import pandas as pd

df = pd.read_csv('Downloads/ex2data2.csv', sep = ',')
df.head()


# In[11]:


X_train = df.iloc[:,:-1]
Y_train = df.iloc[:, -1]

X_train = np.array(X_train)
Y_train = np.array(Y_train)
Y_train = Y_train.reshape(Y_train.shape[0], 1)

print(X_train.shape)
print(Y_train.shape)


# In[12]:


mean = np.mean(X_train, axis = 0)
variance = np.var(X_train, axis = 0)

X_train = np.divide((X_train - mean), variance)


# In[13]:


print(X_train[0])


# In[14]:


Y_train = Y_train - 1
# Changing label 1 to 0 and label 2 to 1


# In[21]:


# Split the data into test and train sets
from sklearn.utils import shuffle

X_train, Y_train = shuffle(X_train, Y_train)

X_test = X_train[85:,:]
Y_test = Y_train[85:,:]

X_train_ = X_train[:250,:]
Y_train_ = Y_train[:250,:]

print(X_train_.shape)
print(Y_train_.shape)
print(X_test.shape)
print(Y_test.shape)


# In[22]:


X_train_ = X_train_.reshape(2, 117)
Y_train_ = Y_train_.reshape(1, 117)
X_test  = X_test.reshape(2, 32)
Y_test  = Y_test.reshape(1, 32)


# In[151]:


m = model()
m.add_layers([2, 8, 8, 1])

m.train(X_train_, Y_train_, alpha = 0.001, decay_rate = 0.90, decay_iter = 15, stop_decay_counter = 100, 
        iterations = 12500, verbose = False, lam = 2)


# In[152]:


m.plot_cost()


# In[153]:


m.plot_acc()


# In[154]:


print('Test set acc = ', m.evaluate(X_test, Y_test))

