'''
Created on Jul 31, 2018

@author: abhinav.jhanwar
'''

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.utils import resample
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

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

# hence this model is only predicting 0, which means it's completely ignoring the
# minority class in favor of the majority class.

'''Up-sample Minority Class'''

# Separate majority and minority classes
df_majority = df[df.balance==0]
df_minority = df[df.balance==1]

# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=576,    # to match majority class
                                 random_state=123) # reproducible results

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

# Display new class counts
df_upsampled.balance.value_counts()

# Separate input features (X) and target variable (y)
y = df_upsampled.balance
X = df_upsampled.drop('balance', axis=1)
 
# Train model
clf_1 = LogisticRegression().fit(X, y)
 
# Predict on training set
pred_y_1 = clf_1.predict(X)
 
# Is our model still predicting just one class?
print( np.unique( pred_y_1 ) )
# [0 1]
 
# How's our accuracy?
print( accuracy_score(y, pred_y_1) )
# 0.513888888889

print(confusion_matrix(y, pred_y_1))



'''Down-sample Majority Class'''
# Separate majority and minority classes
df_majority = df[df.balance==0]
df_minority = df[df.balance==1]
 
# Downsample majority class
df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=49,     # to match minority class
                                 random_state=123) # reproducible results
 
# Combine minority class with downsampled majority class
df_downsampled = pd.concat([df_majority_downsampled, df_minority])
 
# Display new class counts
df_downsampled.balance.value_counts()
# 1    49
# 0    49
# Name: balance, dtype: int64

# Separate input features (X) and target variable (y)
y = df_downsampled.balance
X = df_downsampled.drop('balance', axis=1)
 
# Train model
clf_2 = LogisticRegression().fit(X, y)
 
# Predict on training set
pred_y_2 = clf_2.predict(X)
 
# Is our model still predicting just one class?
print( np.unique( pred_y_2 ) )
# [0 1]
 
# How's our accuracy?
print( accuracy_score(y, pred_y_2) )
# 0.581632653061

print(confusion_matrix(y, pred_y_2))


'''Change Your Performance Metric'''
# if roc score comes under 0.5 i.e. you need to invert the predictions> change p[1] to p[0]
# because Scikit-Learn is misinterpreting the positive class. AUROC should be >= 0.5.

# Predict class probabilities
prob_y_2 = clf_2.predict_proba(X)
 
# Keep only the positive class
prob_y_2 = [p[1] for p in prob_y_2]
 
prob_y_2[:5] # Example
# [0.45419197226479618,
#  0.48205962213283882,
#  0.46862327066392456,
#  0.47868378832689096,
#  0.58143856820159667]

# performance of downsampled data on AUROC curve
print( roc_auc_score(y, prob_y_2) )
# 0.5680966264056644

# performance of original data on AUROC curve
prob_y_0 = clf_0.predict_proba(X)
prob_y_0 = [p[1] for p in prob_y_0]
 
print( roc_auc_score(y, prob_y_0) )
# 0.530718537415


'''Penalize Algorithms (Cost-Sensitive Training)'''

# Separate input features (X) and target variable (y)
y = df.balance
X = df.drop('balance', axis=1)
 
# Train model
clf_3 = SVC(kernel='linear', 
            class_weight='balanced', # penalize
            probability=True)
 
clf_3.fit(X, y)
 
# Predict on training set
pred_y_3 = clf_3.predict(X)
 
# Is our model still predicting just one class?
print( np.unique( pred_y_3 ) )
# [0 1]
 
# How's our accuracy?
print( accuracy_score(y, pred_y_3) )
# 0.688
 
# What about AUROC?
prob_y_3 = clf_3.predict_proba(X)
prob_y_3 = [p[0] for p in prob_y_3]
print( roc_auc_score(y, prob_y_3) )
# 0.5305236678


'''Use Tree-Based Algorithms'''

'''In modern applied machine learning, tree ensembles 
(Random Forests, Gradient Boosted Trees, etc.) 
almost always outperform singular decision trees, so we'll jump right into those'''


# Separate input features (X) and target variable (y)
y = df.balance
X = df.drop('balance', axis=1)
 
# Train model
clf_4 = RandomForestClassifier()
clf_4.fit(X, y)
 
# Predict on training set
pred_y_4 = clf_4.predict(X)

# Is our model still predicting just one class?
print( np.unique( pred_y_4 ) )
# [0 1]
 
# How's our accuracy?
print( accuracy_score(y, pred_y_4) )
# 0.9744
 
# What about AUROC?
prob_y_4 = clf_4.predict_proba(X)
prob_y_4 = [p[1] for p in prob_y_4]
print( roc_auc_score(y, prob_y_4) )
# 0.999078798186

print(confusion_matrix(y, pred_y_4))

'''True positive rate (TPR), aka. sensitivity, hit rate, and recall, which is defined 
as TP/TP+FN. 
False positive rate (FPR), aka. fall-out, which is defined as FP/FP+TN'''
