#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import os
print(os.listdir("../IrfanFath/Documents"))


# In[2]:


dfAll = pd.read_csv("../IrfanFath/Documents/Dish.csv")
df = dfAll[((dfAll.id == 514319) | (dfAll.id == 514319))]


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


from sklearn.model_selection import train_test_split

X = df.drop(["id"], axis=1)
Y = df.menu_id
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)


# In[ ]:


plt.figure(figsize=(8,8))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.axis('off')
    plt.imshow(x_train.head().values[i].reshape(28,28), cmap='gray', interpolation='none')


# In[6]:


x_train = x_train.values.T
y_train = y_train.values.reshape(8400,1).T
x_test = x_test.values.T
y_test = y_test.values.reshape(3600,1).T


# In[ ]:


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def buildClassifier():
    classifier = Sequential()
    classifier.add(Dense(units=8, kernel_initializer="uniform", activation="relu", input_dim=x_train.shape[0])) # Hidden Layer 1 with 8 nodes
    classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu"))  # Hidden Layer 2 with 6 nodes
    classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid")) # Output Layer
    classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
    return classifier


classifier = KerasClassifier(build_fn=buildClassifier, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = x_train.T, y = y_train.T, cv=3)
mean = accuracies.mean()
variance = accuracies.std()

