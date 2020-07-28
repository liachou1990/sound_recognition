#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import keras
from keras import layers
from keras import models
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.regularizers import l1
from keras import metrics


# In[3]:


# load 'feat.npy'
feat = np.load('feat.npy',allow_pickle=True)


# In[4]:


# load 'path.npy'
path = np.load('path.npy', allow_pickle=True)


# In[5]:


# load 'train.csv'
train = pd.read_csv('train.csv')


# In[6]:


# load 'test.csv'
test = pd.read_csv('test.csv')


# In[7]:


# create dictionary using path values as keys and feat values as values
dictionary = {}
for key,val in zip(path,feat):
    dictionary[key] = val


# In[8]:


# match train set with dictionary's items
y_big_train = []
x_big_train = []
for i in train.values:
    if i[0] in dictionary.keys():
        y_big_train.append(i[1])
        x_big_train.append(dictionary[i[0]])
y_big_train = np.asarray(y_big_train)
x_big_train = np.asarray(x_big_train)


# In[9]:


# match test set with dictionary's items
x_test = []
for i in test.values:
    if i[0] in dictionary.keys():
        x_test.append(dictionary[i[0]])
x_test = np.asarray(x_test)


# In[10]:


# applying the StandardScaler from sklearn to preprocess the train - test data sets
scaler = StandardScaler()

for i in range(94824):
    x_big_train[i] = scaler.fit_transform(x_big_train[i])
    
for i in range(11005):
    x_test[i] = scaler.fit_transform(x_test[i])   


# In[11]:


# checking the highest length of the first dimension of the train data
counter=0
for i in range(x_big_train.shape[0]):
    if x_big_train[i].shape[0] > counter:
        counter = x_big_train[i].shape[0]

#Create a zero matrix for the input of train data
train_data = np.zeros((94824,99,13))
#Concatenating the values from the train data to the zero matrix
for i in range(94824):
    for j in range(x_big_train[i].shape[0]):
        for z in range(x_big_train[i].shape[1]):
            train_data[i][j][z] = train_data[i][j][z] + x_big_train[i][j][z]


# In[12]:


# checking the highest length of the first dimension of the test data
counter=0
for i in range(x_test.shape[0]):
    if x_test[i].shape[0] > counter:
        counter = x_test[i].shape[0]      

#Create a zero matrix for the input of spoken data
test_data = np.zeros((11005,99,13))
#Concatenating the values from the spoken data to the zero matrix
for i in range(11005):
    for j in range(x_test[i].shape[0]):
        for z in range(x_test[i].shape[1]):
            test_data[i][j][z] = test_data[i][j][z] + x_test[i][j][z]


# In[13]:


# train - validation sets split
x_train, x_val, y_train, y_val = train_test_split(train_data, y_big_train, test_size=0.2, random_state=666, shuffle=True)


# In[14]:


# transform the labels for the processing of the model
lb = LabelEncoder()
big_train_labels = lb.fit_transform(y_big_train)
train_labels = lb.fit_transform(y_train)
val_labels = lb.fit_transform(y_val)


# In[15]:


# transform the above labels to binary in order to fit them in the network
big_train_labels = tf.keras.utils.to_categorical(big_train_labels, num_classes=None, dtype='float32')
train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=None, dtype='float32')
val_labels = tf.keras.utils.to_categorical(val_labels, num_classes=None, dtype='float32')


# In[16]:


# apply the 1-D Convolutional Neural Network
n_timesteps  = x_train.shape[1]
n_features =  x_train.shape[2] 
n_outputs = 35

model = models.Sequential()
model.add(layers.Conv1D(filters=100, kernel_size=10, activation='relu', input_shape=(n_timesteps,n_features)))
model.add(layers.MaxPooling1D())
model.add(layers.BatchNormalization())
model.add(layers.Conv1D(100, 10, activation='relu'))
model.add(layers.MaxPooling1D())
model.add(layers.BatchNormalization())
model.add(layers.Conv1D(100, 10, activation='relu'))
model.add(layers.MaxPooling1D())
model.add(layers.BatchNormalization())
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(n_outputs, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# In[17]:


# fit using train set
model.fit(x_train, train_labels, epochs=20)


# In[18]:


# evaluate using validation set
accuracy = model.evaluate(x_val, val_labels, batch_size=32)
print(model.metrics_names)
print(accuracy)


# In[19]:


# predict using test set
test = model.predict(test_data)
test = np.argmax(np.round(test),axis=1)
test = lb.inverse_transform(test)


# In[20]:


# add the words' column to test set - create csv file
new = pd.read_csv('test.csv', delimiter=',')
new['word'] = test


# In[21]:


# save the csv file
new = new.to_csv(r'C:\Users\Gebruiker\Downloads\ml_challenge\result.csv', index=False)

