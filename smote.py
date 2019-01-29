'''
Created on Jul 31, 2018

@author: abhinav.jhanwar
'''

''' 
SMOTE (Synthetic Minority Over-sampling TEchnique)
'''

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from imblearn.over_sampling import SMOTE

df = pd.read_csv("balance-scale.data", names=['balance', 'var1', 'var2', 'var3', 'var4'])

df.head()

# check the distribution of df
df['balance'].value_counts()

# transform into binary classification
# balance df = 1 and imbalance df = 0
df.balance = [1 if b=='B' else 0 for b in df.balance]

# Separate input features (X) and target variable (y)
y = df.balance
X = df.drop('balance', axis=1)
 
# Train model
clf_0 = LogisticRegression().fit(X, y)
 
# Predict on training set
pred_y_0 = clf_0.predict(X)

# How's the accuracy?
print( accuracy_score(pred_y_0, y) )

# Should we be excited?
print( np.unique( pred_y_0 ) )

print(confusion_matrix(y, pred_y_0))


''' Applying SMOTE'''

sm = SMOTE(random_state=12, ratio = 'auto')
x_res, y_res = sm.fit_sample(X,y)

print( y.value_counts(), np.bincount(y_res))

# Train model
clf_1 = LogisticRegression().fit(x_res, y_res)
 
# Predict on training set
pred_y_1 = clf_1.predict(x_res)

# How's the accuracy?
print( accuracy_score(pred_y_1, y_res) )

# Should we be excited?
print( np.unique( pred_y_1 ) )

print(confusion_matrix(y_res, pred_y_1))