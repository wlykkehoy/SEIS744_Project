#!/usr/bin/env python
# coding: utf-8

# ### Load and Run Decision TRee Model
# 
# Test code for loading the Decision Tree model pipeline (scaler + dec tree) and running some predictions.

# In[1]:


# Import libraries we will be using

import sys
import numpy as np
import pandas as pd

import sklearn.metrics as skl_met

import pickle


# In[2]:


# Always a good idea to dump verions of key libraries
print('python version:    ', sys.version)
print('numpy version:     ', np.__version__)
print('pandas version:    ', pd.__version__)


# In[3]:


# Load the pickled model
model = pickle.load(open('tree_pipeline.pkl', 'rb'))


# In[4]:


# Let's test the pipeline with some randomly selected values from the 
#   original data file
X_test = [[0.074531], [0.847295], [0.194172], [0.649200], [0.058840], [0.207901]]
y_test_class = [0, 2, 1, 2, 0, 1]

yhat_test_class = model.predict(X_test)

print('Confusion Matrix:\n', skl_met.confusion_matrix(y_test_class, yhat_test_class))

